import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from .ProblemType import RegressionNN, ClassificationNN
from exp.DFRdatasets.data_utils import EarlyStoppingScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau



class _BaseNN(nn.Module):
    def __init__(self, mode, loss_criteria=None, **kwargs):
        super(_BaseNN, self).__init__()
        self.mode = mode
        if self.mode == 'regression':
            self.problem_type_obj = RegressionNN(loss_criteria=loss_criteria)
        else:
            self.problem_type_obj = ClassificationNN(loss_criteria=loss_criteria)

    def train_eval_loader(self, loader, cuda=False):
        def _train_callback(the_loss):
            # zero the parameter gradients
            self.optimizer.zero_grad()
            the_loss.backward()
            self.optimizer.step()

        self.train()
        return self._evaluate_loader(loader, cuda, callback=_train_callback)

    def test_eval_loader(self, loader, cuda=False):
        self.eval()
        return self._evaluate_loader(loader, cuda)

    def _evaluate_loader(self, loader, cuda=False, callback=None):
        for i, data in enumerate(loader):
            # get the inputs
            inputs, targets = data

            inputs = Variable(inputs)
            targets = Variable(targets)
            if cuda:
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)

            outputs = self.forward(inputs)
            loss = self.problem_type_obj.eval_criteria(outputs, targets)

            if callback is not None:
                callback(loss)

        return self.problem_type_obj.ret_metrics()

    def verbose_print(self, the_str):
        if self.verbose > 0:
            print(the_str)

    @staticmethod
    def dict_to_str(the_dict):
        str_arr = []
        for k in the_dict:
            str_arr.append('%s: %.3f' % (k, the_dict[k]))
        return ', '.join(str_arr)

    def fit(self, train_loader, val_loader=None, test_loader=None, epochs=1000,
            epoch_print=1, weight_lr=1e-3, lookahead=10, time_budget=None, lr_patience=10,
            save_freq=None, cuda=False, verbose=0, weight_decay=0., **kwargs):
        if cuda:
            self.cuda()

        self.verbose = verbose

        if not hasattr(self, 'optimizer'):
            betas = [0.9, 0.999]
            if 'beta1' in kwargs:
                betas[0] = kwargs['beta1']
            self.optimizer = optim.Adam(self.parameters(), lr=weight_lr,
                                        weight_decay=weight_decay, betas=tuple(betas))

        start_time = time.time()

        # LR schedule and lookahead early stopping
        scheduler = ReduceLROnPlateau(self.optimizer, patience=lr_patience,
                                      factor=0.2, min_lr=5E-7, verbose=True)
        if lookahead is not None:
            early_stop = EarlyStoppingScheduler(lookahead=lookahead)

        for epoch in range(epochs):
            train_metrics = self.train_eval_loader(train_loader, cuda=cuda)
            if val_loader is not None:
                val_metrics = self.test_eval_loader(val_loader, cuda=cuda)

            if epoch % epoch_print == (epoch_print - 1):
                self.verbose_print('epoch %d: train => %s, val => %s' \
                       % (epoch, self.dict_to_str(train_metrics),
                          self.dict_to_str(val_metrics)))

            val_loss = val_metrics['loss']

            if save_freq is not None and epoch % save_freq == (save_freq - 1):
                pass
                # self.save_net(epoch)

            # Reduce learning rate if epoch loss is not improving
            scheduler.step(val_loss)
            # Early Stopping
            if lookahead is not None and early_stop.is_early_stop(val_loss):
                break

            if time_budget is not None and time.time() - start_time > time_budget:
                self.verbose_print('Exceeds time budget %d seconds! '
                                   'Exit training.' % time_budget)
                break

        self.verbose_print('Finished Training! Spend {} time'.format(
            time.time() - start_time))

        if test_loader is not None:
            self.verbose_print('Evaluating the test log likelihood...')
            test_metrics = self.test_eval_loader(test_loader, cuda=cuda)
            self.verbose_print('test: %s' % (str(test_metrics)))

        if save_freq is not None:
            pass
            # self.save_net(epoch)
            # self.record_final_result(**locals())
