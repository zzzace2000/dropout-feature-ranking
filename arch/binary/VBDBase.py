from __future__ import print_function
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import pandas as pd


class VBDBase(nn.Module):
    def __init__(self, dim_input, dim_output, thresh=0, ard_init=1,
                 anneal=1.05, anneal_max=100, rw_max=20, name=None):
        super(VBDBase, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.logit_p = Parameter(torch.Tensor(dim_input, dim_output))
        self.logit_p.data.fill_(ard_init)

        self.thresh = thresh
        self.ard_init = ard_init

        self.anneal = anneal
        self.anneal_max = anneal_max
        self.rw_max = rw_max
        self.reached_max = False
        self.optimizer = None

        if name is None:
            self.hash = ''.join([chr(random.randint(97, 122)) for _ in range(3)])
        else:
            self.hash = name

    @staticmethod
    def clip(mtx, to=5):
        mtx.data[mtx.data > to] = to
        mtx.data[mtx.data < -to] = -to
        return mtx

    def anneal_policy(self, epoch):
        if self.reached_max:
            return self.anneal_max

        anneal_val = self.anneal ** epoch
        if anneal_val > self.anneal_max:
            self.reached_max = True
            return self.anneal_max

        return anneal_val

    def sgvloss(self, outputs, targets, rw, num_samples):
        raise NotImplementedError

    def eval_reg(self):
        raise NotImplementedError

    def rw_policy(self, epoch):
        if epoch > self.rw_max:
            return 1.

        return epoch * 1.0 / self.rw_max

    def get_sparsity(self, **kwargs):
        return '%.3f(threshold %.1f)' % ((self.logit_p.data < self.thresh).sum() * 1.0
                                        / torch.numel(self.logit_p.data), self.thresh)

    def get_alpha_range(self):
        logit_p = self.clip(self.logit_p)
        return '%.2f, %.2f' % (logit_p.data.min(), logit_p.data.max())

    def eval_criteria(self, outputs, targets):
        raise NotImplementedError

    def get_val_criteria(self, loader, cuda=False):
        print_statistics = [0.]
        for i, data in enumerate(loader):
            # get the inputs
            inputs, targets = data

            inputs = Variable(inputs)
            targets = Variable(targets)
            if cuda:
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)

            outputs = self.forward(inputs, testing=True)

            acc = self.eval_criteria(outputs, targets)

            print_statistics[0] += acc
        return print_statistics[0] * 1.0 / loader.dataset.data_tensor.size(0)

    def fit(self, data_loader, valloader, testloader=None, stochastic=False, max_iter=1000,
            batch_print=10, epoch_print=1,
            weight_lr=1e-3, logitp_lr=1e-3, pretrain=False, train_clip=False, lookahead=10,
            time_budget=None, lr_patience=10, save_freq=None, cuda=False, decrease_logitp_lr=True):
        if cuda:
            self.cuda()

        if pretrain:
            logitp_lr = 0.

        other_params = [p for name, p in self.named_parameters() if name != 'logit_p']
        if self.optimizer is None:
            self.optimizer = optim.Adam([{'params': other_params},
                                         {'params': [self.logit_p], 'lr': logitp_lr}],
                                        lr=weight_lr)
        else:
            self.optimizer.param_groups[0]['lr'] = weight_lr
            self.optimizer.param_groups[1]['lr'] = logitp_lr

        def reduce_lr(ratio=3., min_lr=5E-6):
            for i, param_group in enumerate(self.optimizer.param_groups):
                if not decrease_logitp_lr and i == 1:
                    continue
                old_lr = float(param_group['lr'])
                new_lr = old_lr / ratio
                if new_lr < min_lr:
                    new_lr = min_lr
                param_group['lr'] = new_lr

        start_time = time.time()

        min_val_loss = np.inf
        min_epoch = 0
        lr_counter = lr_patience
        N = data_loader.dataset.data_tensor.size(0)

        val_loss = []
        train_pred_loss = []
        train_reg_loss = []
        for epoch in range(max_iter):
            print_statistics = [0., 0., 0., 0.]

            epoch_st_time = time.time()
            total_batch = len(data_loader)
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
                outputs = self.forward(inputs, epoch=epoch, stochastic=stochastic, testing=pretrain,
                                       train_clip=train_clip)

                sgv_loss, pred_loss, reg_loss = self.sgvloss(outputs, targets, rw=self.rw_policy(epoch),
                                                             num_samples=N)
                the_loss = pred_loss if pretrain else sgv_loss
                the_loss.backward()
                self.optimizer.step()

                acc = self.eval_criteria(outputs, targets)

                print_statistics[0] += sgv_loss.data[0]
                print_statistics[2] += pred_loss.data[0]
                print_statistics[3] += reg_loss.data[0]
                print_statistics[1] += acc
                num += inputs.size(0)

                if batch_idx % batch_print == (batch_print - 1):
                    print('epoch %d [%d / %d]: loss %.5f (%.5f, %.5f)' % \
                          (epoch, batch_idx, total_batch, print_statistics[0] / num,
                           print_statistics[2] / num, print_statistics[3] / num))

            val_criteria = self.get_val_criteria(valloader, cuda=cuda)

            if epoch % epoch_print == (epoch_print - 1):
                print('epoch: %d, val: %.3f, train: %.3f, loss: %.5f (%.5f, %.5f), ' \
                      'sparsity: %s, range: %s, (%.1f secs)' % \
                      (epoch, val_criteria, print_statistics[1] / N, print_statistics[0] / N,
                       print_statistics[2] / N, print_statistics[3] / N,
                       self.get_sparsity(), self.get_alpha_range(), time.time() - epoch_st_time))

            val_loss.append(val_criteria)
            train_pred_loss.append(print_statistics[2] / N)
            train_reg_loss.append(print_statistics[3] / N)

            if min_val_loss > val_criteria:
                min_val_loss = val_criteria
                min_epoch = epoch
                lr_counter = lr_patience
            else:
                if epoch - min_epoch > lookahead:
                    break

                lr_counter -= 1
                if lr_counter == 0:
                    print('reduce learning rate!')
                    reduce_lr()
                    lr_counter = lr_patience

            if save_freq is not None and epoch % save_freq == (save_freq - 1):
                self.save_net(epoch)

            if time_budget is not None and time.time() - start_time > time_budget:
                print('Exceeds time budget %d seconds! Exit training.' % time_budget)
                break

        print('Finished Training')

        if pretrain:
            return

        test_loss = None
        if testloader is not None:
            print('Evaluating the test log likelihood...')
            test_loss = self.get_val_criteria(testloader, cuda=cuda)
            print('test llk: %.3f, sparsity: %s' % (test_loss, self.get_sparsity()))

        if save_freq is not None:
            self.save_net(epoch)
            self.record_final_result(**locals())
        return

    def save_net(self, epoch=1):
        if not os.path.exists('model'):
            os.mkdir('model')

        fname = sys.argv[0].split('/')[-1][:-3]
        folder_name = 'model/%s-%s-%s' % (self.__class__.__name__, self.hash, fname)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        name = '%s/%s' % (folder_name, epoch)
        print(('save model: ' + name))
        torch.save(self, name)

    def record_final_result(real_self, **kwargs):
        if not os.path.exists('results'):
            os.mkdir('results')

        fname = sys.argv[0].split('/')[-1][:-3]
        folder = 'results/%s-%s-%s' % (real_self.__class__.__name__, real_self.hash, fname)
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Save Excel files
        filename = '%s/exp.tsv' % folder

        # Exp settings
        run_headers = ['lookahead', 'lr_patience', 'weight_lr', 'logitp_lr', 'decrease_logitp_lr',
                       'stochastic']
        run_values = [str(kwargs[h]) for h in run_headers]
        net_headers = ['ard_init', 'anneal', 'anneal_max', 'rw_max']
        net_values = [str(getattr(real_self, h)) for h in net_headers]

        # Exp result
        exp_headers = ['name', 'range', 'sparsity', 'test_loss', 'min_val_loss', 'min_train_pred',
                       'epoch']
        exp_vals = [str(real_self.hash), real_self.get_alpha_range(), real_self.get_sparsity(),
                    str(kwargs['test_loss'])] + \
                   [str(min(kwargs[h])) for h in ['val_loss', 'train_pred_loss']] + \
                   [str(kwargs['epoch'])]

        # Custom settings
        custum_header, custom_vals = [], []
        if hasattr(real_self, 'get_custom_settings'):
            custum_header, custom_vals = real_self.get_custom_settings()

        with open(filename, 'w') as op:
            print('\t'.join(exp_headers + run_headers + custum_header + net_headers), file=op)
            print('\t'.join(exp_vals + run_values + custom_vals + net_values), file=op)
        print('save exp:', filename)

        # Save Figs:
        filename = '%s/loss.png' % folder

        data = {
                'val_loss': kwargs['val_loss'],
                'train_pred_loss': kwargs['train_pred_loss'],
                'train_reg_loss': kwargs['train_reg_loss'],
               }
        df = pd.DataFrame.from_dict(data)
        ax = df.plot()
        ax.set_xlabel('epochs')
        ax.set_ylabel('NLL Loss (nat)')
        plt.savefig(filename)
        print('save figure:', filename)


