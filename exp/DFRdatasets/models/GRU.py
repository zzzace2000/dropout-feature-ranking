import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


class GRU(nn.Module):
    '''
    A many-to-one implementation of GRU.
    '''
    def __init__(self, input_size, hidden_size, weight_1_class=1, append_mask=True):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.append_mask = append_mask
        self.weight_1_class = weight_1_class
        self.weight = nn.Parameter(torch.FloatTensor([1, self.weight_1_class]),
                                   requires_grad=False)

        self.output_dropout = nn.Dropout(0.5)
        self.output_batchnorm = nn.BatchNorm1d(hidden_size)
        if append_mask:
            input_size *= 2
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, x, seq_lengths):
        # Sort them
        sort_idx = np.argsort(-seq_lengths)
        x_new = x[x.data.new(sort_idx).long()]
        sort_lens = seq_lengths[sort_idx]
        unsort_idx = np.argsort(sort_idx)

        # Append the missing features
        missing = x_new.data.new(*x_new.size()).fill_(0.)
        missing[x_new.data == 0.] = 1.
        x_new = torch.cat([x_new, Variable(missing)], dim=2)

        # Do dropout myself. Sample a B x 1 x D bernoulli
        if self.training:
            dropout_mask = x.data.new(x_new.size(0), 1, x_new.size(2))\
                               .fill_(0.3).bernoulli_() * 1.0 / 0.7
            x_new = x_new * Variable(dropout_mask)

        # batch_in = Variable(x_new, volatile=volatile)
        pack = pack_padded_sequence(x_new, sort_lens, batch_first=True)

        _, hn = self.gru(pack)

        last_hidden = hn[-1]
        last_hidden = self.output_batchnorm(last_hidden)
        last_hidden = self.output_dropout(last_hidden)
        out1 = self.linear(last_hidden)

        # Sort it back
        out1 = out1[x.data.new(unsort_idx).long()]
        return out1

    def loss_fn(self, outputs, targets):
        # Use a multi-class method to solve the class weight problem
        loss = nn.CrossEntropyLoss(weight=self.weight.data)(outputs, targets)
        return loss

    def eval_loader(self, loader, cuda=False, callback=None, eval_only=False):
        original_state = self.training
        if eval_only:
            self.eval()

        num_correct = 0.
        total_loss = 0.
        num_instances = 0.

        for batch_idx, (x, lengths, y, total_batches) in enumerate(loader):
            if cuda:
                x = x.cuda(async=True)
                y = y.cuda(async=True)

            x = Variable(x, volatile=eval_only)
            y = Variable(y, volatile=eval_only)

            # zero the parameter gradients
            outputs = self.forward(x, lengths)
            the_loss = self.loss_fn(outputs, y)

            _, pred_class = outputs.data.max(dim=1)
            num_correct += pred_class.eq(y.data).sum()
            total_loss += the_loss.data[0] * outputs.size(0)
            num_instances += outputs.size(0)

            if callback is not None:
                avg_loss, acc = total_loss / num_instances, num_correct / num_instances
                callback(batch_idx, total_batches, the_loss, avg_loss, acc)

        if eval_only:
            self.train(original_state)
        return total_loss / num_instances, num_correct / num_instances

    def eval_auroc_aupr(self, loader, cuda=False, input_process_callback=None):
        original_state = self.training
        self.eval()

        prob_measure = []
        ys = []
        for batch_idx, (x, lengths, y, total_batches) in enumerate(loader):
            if cuda:
                x = x.cuda(async=True)
                y = y.cuda(async=True)

            x = Variable(x, volatile=True)
            if input_process_callback is not None:
                x = input_process_callback(x)

            outputs = self.forward(x, lengths)
            probs = F.softmax(outputs)
            prob_measure.append(probs.data[:, 1])
            ys.append(y)

        prob_measure = torch.cat(prob_measure, dim=0)
        ys = torch.cat(ys, dim=0)

        auroc = roc_auc_score(ys.cpu().numpy(), prob_measure.cpu().numpy(), average='macro')
        aupr = average_precision_score(ys.cpu().numpy(), prob_measure.cpu().numpy(), average='macro')
        self.train(original_state)
        return auroc, aupr

if __name__ == '__main__':
    import copy
    model = GRU(64, 37)
    new_model = copy.deepcopy(model)
    print(new_model.cpu())