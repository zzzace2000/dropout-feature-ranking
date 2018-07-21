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


class DFSNet(DNetBase):
    def __init__(self, param_size, trained_classifier, loss_criteria=nn.CrossEntropyLoss(),
                 lr=0.01, l1_reg_coef=1., cuda_enabled=True, verbose=0):
        super(DFSNet, self).__init__(trained_classifier, loss_criteria,
                                     lr, reg_coef=1., rw_max=1, cuda_enabled=cuda_enabled,
                                     verbose=verbose, estop_num=None, clip_max=10000, clip_min=-10000)
        self.param_size = param_size
        self.map_weights = Parameter(torch.Tensor(*param_size))
        self.l1_reg_coef = l1_reg_coef

        for param in self.trained_classifier.parameters():
            param.requires_grad = True

    def param_name(self):
        return 'map_weights'

    def sgvloss(self, outputs, targets, rw=1.0):
        avg_pred_loss = self.loss_criteria(outputs, targets)
        reg_loss = self.eval_reg().sum()

        return avg_pred_loss + reg_loss, avg_pred_loss, reg_loss

    def eval_reg(self):
        reg = self.l1_reg_coef * torch.abs(self.map_weights)
        return reg

    def my_parameters(self):
        arr = [self.map_weights] + list(self.trained_classifier.parameters())
        return arr

    def initialize(self, train_loader, epochs):
        super(DFSNet, self).initialize(train_loader, epochs)
        self.map_weights.data.fill_(1.)

    def forward(self, input):
        weights = self.map_weights.unsqueeze(0).expand_as(input)
        noised_input = input * weights

        return self.trained_classifier(noised_input)


class BinaryDFSNet(DFSNet):
    def eval_criteria(self, outputs, targets):
        outputs = outputs.round()
        correct_count = outputs.eq(targets).sum().data[0]
        return correct_count
