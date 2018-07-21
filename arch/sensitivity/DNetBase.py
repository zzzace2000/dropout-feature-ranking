import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import torch.optim as optim
import time
from torch.optim.lr_scheduler import LambdaLR
import visdom
import numpy as np
from exp.general_utils import Timer


class DNetBase(nn.Module):
    def __init__(self, trained_classifier, loss_criteria=nn.CrossEntropyLoss(),
                 lr=0.01, reg_coef=1., rw_max=30, cuda_enabled=False,
                 verbose=0, estop_num=None, clip_max=8., clip_min=-8., visdom_enabled=False, **kwargs):
        super(DNetBase, self).__init__()

        self.trained_classifier = trained_classifier
        self.loss_criteria = loss_criteria
        self.lr = lr
        self.reg_coef = reg_coef
        self.rw_max = rw_max
        self.cuda_enabled = cuda_enabled
        self.verbose = verbose
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.estop_num = estop_num

        # Remove the gradient in pretrained model
        self.trained_classifier.eval()
        for param in trained_classifier.parameters():
            param.requires_grad = False

        self.callbacks = []

        self.lowest_metric = None
        self.bad_epoch = None
        self.visdom_enabled = visdom_enabled
        if self.visdom_enabled:
            self.vis = visdom.Visdom()

    def my_parameters(self):
        return [self.get_param()]

    def initialize(self, train_loader, epochs):
        self.optimizer = optim.Adam(self.my_parameters(), lr=self.lr)

        lr_policy = lambda epoch: (1 - epoch * 1.0 / epochs)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_policy)

        if self.cuda_enabled:
            self.cuda()

        self.loss_arr = []
        self.loss_win = None
        self.loss_epochs = []
        self.dropout_win = None

    def param_name(self):
        raise BaseException('You need to define which param you want to monitor')

    def get_param(self):
        return self.__getattr__(self.param_name())

    def clip(self, mtx):
        return mtx.clamp(min=self.clip_min, max=self.clip_max)

    def get_params_range(self):
        return '(%.2f, %.2f, %.2f)' % (self.get_param().data.min(),
                                       self.get_param().data.mean(),
                                       self.get_param().data.max())

    def sgvloss(self, outputs, targets, rw=1.0):
        avg_pred_loss = self.loss_criteria(outputs, targets)
        reg_loss = self.reg_coef * (self.eval_reg().sum())

        return avg_pred_loss + rw * reg_loss, avg_pred_loss, rw * reg_loss

    def rw_policy(self, epoch):
        # Linearly increase rw from 0 to self.rw_max
        if epoch > self.rw_max:
            return 1.
        return epoch * 1.0 / self.rw_max

    def eval_criteria(self, outputs, targets):
        '''
        Return number of correct classification.
        :param outputs: array size (batch_size, nclass)
        :param targets: array size (batch_size,)
        :return: float
        '''
        targets = targets.long()
        pred_val, pred_pos = outputs.data.max(dim=1)
        acc = (pred_pos == targets.data).sum()
        return acc

    def is_early_stop(self, metric, lookahead):
        if self.bad_epoch is None:
            self.bad_epoch = lookahead

        if self.lowest_metric is None:
            self.lowest_metric = metric
            return False

        if metric <= self.lowest_metric:
            self.lowest_metric = metric
            self.bad_epoch = lookahead
            return False

        self.bad_epoch -= 1
        return self.bad_epoch < 0

    def plot_visdom_loss_curve(self, epoch, title, loss_dict):
        '''
        :param loss_dict: ex => {'sgv_loss': 2.58, 'KL': 0.72}
        '''
        names = list(loss_dict.keys())
        self.loss_arr.append(list(loss_dict.values()))
        self.loss_epochs.append(epoch)

        update = None if not self.loss_win else 'replace'
        opts = dict(xlabel='Epochs', ylabel='Loss', title=title, legend=names)
        self.loss_win = self.vis.line(Y=np.array(self.loss_arr), X=np.array(self.loss_epochs),
                                      update=update, win=self.loss_win, opts=opts)

    def plot_visdom_dropout_heatmap(self, epoch):
        rank = self.get_importance_vector().cpu().numpy()

        # if epoch % 10 == 0:
        #     self.dropout_win = None

        self.dropout_win = self.vis.heatmap(np.flipud(rank), opts=dict(
            xmin=0., xmax=1., title='epochs %d' % epoch), win=self.dropout_win)

    def fit(self, train_loader, epochs, epoch_print=1, val_loader=None, lookahead=None,
            min_epochs=None, input_process_callback=None, vis_title='', **kwargs):
        self.initialize(train_loader, epochs)

        print_statistics = [0., 0., 0., 0.]
        num = 0.
        init_time = time.time()

        for epoch in range(epochs):
            self.scheduler.step()

            epoch_start_time = time.time() # time per epoch speed
            for batch_idx, content in enumerate(train_loader):
                if input_process_callback is not None:
                    inputs, targets = input_process_callback(content)
                else:
                    inputs, targets = content
                    inputs = Variable(inputs)
                    targets = Variable(targets)
                    if self.cuda_enabled:
                        inputs = inputs.cuda(async=True)
                        targets = targets.cuda(async=True)

                self.optimizer.zero_grad()

                corrupted_output = self.forward(inputs)

                sgv_loss, pred_loss, reg_loss = self.sgvloss(corrupted_output, targets,
                                                             rw=self.rw_policy(epoch))
                sgv_loss.backward()
                self.optimizer.step()

                correct_count = self.eval_criteria(corrupted_output, targets)

                print_statistics[0] += sgv_loss.data[0] * targets.size(0)
                print_statistics[1] += correct_count
                print_statistics[2] += pred_loss.data[0] * targets.size(0)
                print_statistics[3] += reg_loss.data[0] * targets.size(0)
                num += targets.size(0)

            opt_str = ''
            if val_loader is not None:
                val_acc, val_loss = self.eval_loader_with_loss_and_acc(val_loader)
                opt_str += 'val_acc: {:.3f} val_loss: {:.2e} '.format(val_acc, val_loss)

            train_loss = print_statistics[0] / num
            train_pred_loss = print_statistics[2] / num
            train_reg_loss = print_statistics[3] / num

            if epoch % epoch_print == (epoch_print - 1) or epoch == 0:
                if len(self.callbacks) > 0:
                    for callback in self.callbacks:
                        opt_str += callback() + ' '

                self.verbose_print('epoch %d / %d: loss %.5f (%.5f, %.5f), acc %.2f, '
                                   'range: %s, takes %.2fs, %s ' % \
                      (epoch, epochs, train_loss,
                       train_pred_loss, train_reg_loss,
                       print_statistics[1] / num, self.get_params_range(),
                       float(time.time() - epoch_start_time), opt_str))
                print_statistics = [0., 0., 0., 0.]
                num = 0.

                if self.visdom_enabled:
                    self.plot_visdom_loss_curve(epoch, title=vis_title, loss_dict={
                        'train loss': train_loss, 'train_pred': train_pred_loss,
                        'train_reg': train_reg_loss})
                    self.plot_visdom_dropout_heatmap(epoch)

            if self.estop_num is not None and self.get_param().data.max() > self.estop_num:
                break

            if lookahead is not None and min_epochs is not None and epoch > min_epochs:
                if val_loader is not None \
                    and self.is_early_stop(val_loss, lookahead):
                    break
                else:
                    if self.is_early_stop(train_loss, lookahead):
                        break

        self.verbose_print('Done! Spend time %.1f seconds.' % (time.time() - init_time))

    def make_predictions(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = Variable(images)
        if self.cuda_enabled:
            images = images.cuda(async=True)

        outputs = self.trained_classifier.forward(images)
        pred_val, pred_pos = outputs.data.max(dim=1)
        return pred_pos

    def register_epoch_callback(self, callback):
        self.callbacks.append(callback)

    def verbose_print(self, the_str):
        if self.verbose > 0:
            print(the_str)

    def eval_loader_with_loss_and_acc(self, loader, cuda=False):
        print_statistics = [0., 0.]
        for i, data in enumerate(loader):
            # get the inputs
            inputs, targets = data

            inputs = Variable(inputs, volatile=True)
            targets = Variable(targets, volatile=True)
            if cuda or self.cuda_enabled:
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)

            outputs = self.forward(inputs)
            sgv_loss, pred_loss, reg_loss = self.sgvloss(outputs, targets)
            correct_count = self.eval_criteria(outputs, targets)

            print_statistics[0] += correct_count
            print_statistics[1] += sgv_loss.data[0] * inputs.size(0)

        avg_acc = print_statistics[0] * 1.0 / loader.dataset.data_tensor.size(0)
        avg_loss = print_statistics[1] * 1.0 / loader.dataset.data_tensor.size(0)
        return avg_acc, avg_loss

    def evaluate_loader(self, loader):
        avg_acc, _ = self.eval_loader_with_loss_and_acc(loader)
        return avg_acc
