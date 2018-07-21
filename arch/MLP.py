import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time
import numpy as np


class MLP(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(dim_input, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, 2)

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def eval_criteria(self, outputs, targets):
        targets = targets.long()
        loss = self.loss(outputs, targets)

        pred_val, pred_pos = outputs.data.max(dim=1)
        acc = (pred_pos == targets.data).sum()
        return loss, acc

    def evaluate_loader(self, loader, cuda=False):
        self.eval()
        print_statistics = [0., 0.]
        for i, data in enumerate(loader):
            # get the inputs
            inputs, targets = data

            inputs = Variable(inputs)
            targets = Variable(targets)
            if cuda:
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)

            outputs = self.forward(inputs)
            loss, correct_count = self.eval_criteria(outputs, targets)

            print_statistics[0] += loss * inputs.size(0)
            print_statistics[1] += correct_count

        avg_loss = print_statistics[0] * 1.0 / loader.dataset.data_tensor.size(0)
        avg_acc = print_statistics[1] * 1.0 / loader.dataset.data_tensor.size(0)
        self.train()
        return avg_loss, avg_acc

    def reduce_lr(self, ratio=2., min_lr=5E-6):
        if not hasattr(self, 'optimizer'):
            raise Exception('Not initialize optimizer')

        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr / ratio
            if new_lr < min_lr:
                new_lr = min_lr
            param_group['lr'] = new_lr

    def verbose_print(self, the_str):
        if self.verbose > 0:
            print(the_str)

    def fit(self, data_loader, valloader, testloader=None, max_iter=1000,
            epoch_print=1, weight_lr=1e-3, lookahead=10, time_budget=None, lr_patience=10,
            save_freq=None, cuda=False, verbose=0, weight_decay=0.):
        if cuda:
            self.cuda()

        self.verbose = verbose

        if not hasattr(self, 'optimizer'):
            self.optimizer = optim.Adam(self.parameters(), lr=weight_lr, weight_decay=weight_decay)

        start_time = time.time()

        min_val_loss = np.inf
        min_epoch = 0
        lr_counter = lr_patience
        N = float(data_loader.dataset.data_tensor.size(0))

        val_losses = []
        train_pred_loss = []
        for epoch in range(max_iter):
            print_statistics = [0., 0.]

            num = 0
            for batch_idx, data in enumerate(data_loader):
                # get the inputs
                inputs, targets = data

                inputs = Variable(inputs)
                targets = Variable(targets)
                if cuda:
                    inputs = inputs.cuda(async=True)
                    targets = targets.cuda(async=True)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)

                the_loss, correct_count = self.eval_criteria(outputs, targets)
                the_loss.backward()
                self.optimizer.step()

                print_statistics[0] += (the_loss.data[0] * inputs.size(0))
                print_statistics[1] += correct_count
                num += inputs.size(0)

            val_loss, val_acc = self.evaluate_loader(valloader, cuda=cuda)

            if epoch % epoch_print == (epoch_print - 1):
                self.verbose_print('epoch: %d, val: %.3f, train: %.3f, val_acc: %.3f, train_acc: %.3f' \
                       % (epoch, val_loss.data[0], print_statistics[0] / N, val_acc,
                          float(print_statistics[1]) / N))

            train_pred_loss.append(print_statistics[0] / N)
            val_losses.append(val_loss)

            if min_val_loss > val_loss.data[0]:
                min_val_loss = val_loss.data[0]
                min_epoch = epoch
                lr_counter = lr_patience
            else:
                if epoch - min_epoch > lookahead:
                    break

                lr_counter -= 1
                if lr_counter == 0:
                    self.verbose_print('reduce learning rate!')
                    self.reduce_lr()
                    lr_counter = lr_patience

            if save_freq is not None and epoch % save_freq == (save_freq - 1):
                pass
                # self.save_net(epoch)

            if time_budget is not None and time.time() - start_time > time_budget:
                self.verbose_print('Exceeds time budget %d seconds! Exit training.' % time_budget)
                break

        self.verbose_print('Finished Training')

        test_loss = None
        if testloader is not None:
            self.verbose_print('Evaluating the test log likelihood...')
            test_loss, test_acc = self.evaluate_loader(testloader, cuda=cuda)
            self.verbose_print('test llk: %.3f, test acc: %.3f' % (test_loss.data[0], test_acc))

        if save_freq is not None:
            pass
            # self.save_net(epoch)
            # self.record_final_result(**locals())


class dropoutMLP(MLP):
    def __init__(self, **kwargs):
        super(dropoutMLP, self).__init__(**kwargs)
        self.drop1 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x


class LinearMLP(MLP):
    def __init__(self, dim_input):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(dim_input, 2)
        self.dim_input = dim_input
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        return x
