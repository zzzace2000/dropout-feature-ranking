import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from ._BaseNN import _BaseNN


class MLP(_BaseNN):
    ''' Regression MLP '''
    def __init__(self, dimensions=[11, 30, 30, 30], mode='regression',
                 loss_criteria=None, **kwargs):
        assert len(dimensions) > 1, 'At least specifies input number and output!'
        super(MLP, self).__init__(mode=mode, loss_criteria=loss_criteria)

        self.ffnn = nn.ModuleList([nn.Linear(dimensions[0], dimensions[1])])
        self.dimensions = dimensions

        for i in range(len(dimensions) - 2):
            self.ffnn.append(nn.ReLU())
            self.ffnn.append(nn.BatchNorm1d(dimensions[i + 1]))
            self.ffnn.append(nn.Dropout(0.5))
            self.ffnn.append(nn.Linear(dimensions[i + 1], dimensions[i + 2]))

    def forward(self, x):
        for layer in self.ffnn:
            x = layer(x)
        return x


class ClassificationMLP(MLP):
    def __init__(self, mode='classification', loss_criteria=None, **kwargs):
        super(ClassificationMLP, self).__init__(
            mode=mode, loss_criteria=loss_criteria, **kwargs)

