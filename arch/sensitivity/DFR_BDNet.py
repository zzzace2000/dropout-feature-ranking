import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import torch.optim as optim
from .DNetBase import DNetBase
from arch.binary.ConcreteNeuron import concrete_dropout_neuron, multiclass_concrete_neuron
from .BDNet import BDNet


L1BDNet = BDNet

class RNNL1BDNet(BDNet):
    def forward(self, inputs):
        x, lengths = inputs
        noised_inputs = self.sampled_noisy_input(x)

        return self.trained_classifier(noised_inputs, lengths)

    def fit(self, train_loader, epochs, epoch_print=1, val_loader=None, lookahead=None,
            min_epochs=None, input_process_callback=None, **kwargs):

        def rnn_callback(contents):
            x, seq_lengths, y, _ = contents
            if self.cuda_enabled:
                x = x.cuda(async=True)
                y = y.cuda(async=True)
            x = Variable(x)
            y = Variable(y)
            return ((x, seq_lengths), y)

        the_callback = rnn_callback
        if input_process_callback is not None:
            the_callback = input_process_callback

        return super(RNNL1BDNet, self).fit(train_loader, epochs, epoch_print, val_loader,
                                           lookahead, min_epochs, the_callback)

    def eval_auroc_aupr(self, loader, cuda=False):
        self.set_eval(True)

        def input_process_callback(x):
            noised_x = self.sampled_noisy_input(x)
            return noised_x

        auroc, aupr = self.trained_classifier.eval_auroc_aupr(
            loader, cuda=cuda, input_process_callback=input_process_callback)
        self.set_eval(False)
        return auroc, aupr


class BinaryMixin:
    def eval_criteria(self, outputs, targets):
        outputs = outputs.round()
        correct_count = outputs.eq(targets).sum().data[0]
        return correct_count


class BinaryL1BDNet(BinaryMixin, BDNet):
    pass


class BinaryBDNet(BinaryMixin, BDNet):
    pass


class TrainWeightsBDNet(BDNet):
    '''
    Used in feature selection. Train the weights and dropout probability.
    Also clip the dropout probability to the threshold
    '''
    def __init__(self, dropout_param_size, trained_classifier, loss_criteria=nn.CrossEntropyLoss(),
                 ard_init=-5, vbd_lr=0.01, weight_lr=0.01, reg_coef=1., rw_max=30, cuda_enabled=True,
                 verbose=0, estop_num=None, clip_max=100, flip_val=0., flip_train=False, thresh=3.,
                 prior_p=0.999):
        super(TrainWeightsBDNet, self).__init__(dropout_param_size, trained_classifier, loss_criteria,
                                         ard_init, vbd_lr, reg_coef, rw_max, cuda_enabled,
                                         verbose, estop_num, clip_max, flip_val, flip_train,
                                                prior_p=prior_p, thresh=thresh)

        for param in self.trained_classifier.parameters():
            param.requires_grad = True

        self.vbd_lr = vbd_lr
        self.weight_lr = weight_lr

        def sparsity():
            return '%.4f (thresh %.1f)' % ((self.logit_p.data > self.thresh).sum() * 1.0
                                            / torch.numel(self.logit_p.data), self.thresh)
        self.register_epoch_callback(sparsity)

    def my_parameters(self):
        arr = [{'params': self.logit_p, 'lr': self.vbd_lr},
               {'params': self.trained_classifier.parameters(), 'lr': self.weight_lr}]
        return arr

    def eval_reg(self):
        logit_p = self.clip(self.logit_p)
        p = torch.sigmoid(logit_p)

        KL = p * (torch.log(p) - np.log(self.prior_p)) \
             + (1 - p) * (torch.log(1 - p) - np.log(1 - self.prior_p))

        KL[logit_p > self.thresh] = 0.
        return KL

    def forward(self, input):
        noised_input = self.sampled_noisy_input(input)

        if self.verbose > 1:
            print('mean_logit_p: %f, Noise mean: %f, Noise var: %f' % (
                self.logit_p.data.mean(), (noised_input - input).data.mean(),
                (noised_input - input).data.var()))

        return self.trained_classifier(noised_input)


class BinaryTrainWeightsBDNet(BinaryMixin, TrainWeightsBDNet):
    pass