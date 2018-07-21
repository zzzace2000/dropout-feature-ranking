import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


class RegressionNN:
    def __init__(self, loss_criteria=None):
        if loss_criteria is None:
            loss_criteria = nn.MSELoss()
        self.loss_criteria = loss_criteria
        self._init_recording()

    def eval_criteria(self, outputs, targets):
        '''
        Evaluate and record the loss. It returns the loss
        :param outputs: Variable. The output of neural network
        :param targets: Variable. The true label
        :return: loss: Variable. The loss needs to be called backward().
        '''
        loss = self.loss_criteria(outputs, targets)
        self.print_statistics[0] += loss.data[0] * outputs.size(0)
        self.num += outputs.size(0)
        return loss

    def _init_recording(self):
        self.print_statistics = [0.]
        self.num = 0

    def ret_metrics(self):
        answer = {'loss': self.print_statistics[0] * 1.0 / self.num}
        self._init_recording()
        return answer


class ClassificationNN:
    def __init__(self, loss_criteria=None):
        if loss_criteria is None:
            loss_criteria = nn.CrossEntropyLoss()
        self.loss_criteria = loss_criteria
        self._init_recording()

    def _eval_criteria(self, outputs, targets):
        targets = targets.long()
        loss = self.loss_criteria(outputs, targets)

        pred_val, pred_pos = outputs.data.max(dim=1)
        correct_counts = (pred_pos == targets.data).sum()
        return loss, correct_counts

    def eval_criteria(self, outputs, targets):
        '''
        Record the loss and output to return metrics.
        :param outputs: Variable. The output of neural network
        :param targets: Variable. The true label
        :param loss: the stuffs returned by function eval_criteria()
        :return:
        '''

        probs = F.softmax(outputs, dim=1)
        self.prob_measure.append(probs.data[:, 1])
        self.ys.append(targets.data)

        loss, correct_counts = self._eval_criteria(outputs, targets)

        self.print_statistics[0] += loss.data[0] * outputs.size(0)
        self.print_statistics[1] += correct_counts
        self.num += outputs.size(0)

        return loss

    def _init_recording(self):
        self.prob_measure, self.ys = [], []
        self.print_statistics = [0., 0.]
        self.num = 0

    def ret_metrics(self):
        avg_loss = self.print_statistics[0] * 1.0 / self.num
        avg_acc = self.print_statistics[1] * 1.0 / self.num

        prob_measure = torch.cat(self.prob_measure, dim=0)
        ys = torch.cat(self.ys, dim=0)
        auroc = roc_auc_score(ys.cpu().numpy(), prob_measure.cpu().numpy(), average='macro')
        aupr = average_precision_score(ys.cpu().numpy(), prob_measure.cpu().numpy(),
                                       average='macro')
        answer = {'loss': avg_loss, 'acc': avg_acc, 'auroc': auroc, 'aupr': aupr}

        self._init_recording()
        return answer
