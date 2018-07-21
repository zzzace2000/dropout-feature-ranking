import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from ._BaseNN import _BaseNN


class ClfLSTM(_BaseNN):
    '''
    A many-to-one implementation of LSTM. Follow the MIMIC3-benchmark
    '''
    def __init__(self, dimensions=[76, 256, 2], **kwargs):
        assert len(dimensions) == 3, 'dimesnions out of range: ' + str(len(dimensions))
        super(ClfLSTM, self).__init__(mode='classification')

        self.input_size = dimensions[0]
        self.hidden_size = dimensions[1]
        self.output_size = dimensions[2]

        self.output_dropout = nn.Dropout(0.5)
        self.output_batchnorm = nn.BatchNorm1d(self.hidden_size)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # Do dropout myself. Sample a B x 1 x D bernoulli
        if self.training:
            dropout_mask = x.data.new(x.size(0), 1, x.size(2)) \
                               .fill_(0.3).bernoulli_() * 1.0 / 0.7
            x = x * Variable(dropout_mask)

        _, hn = self.lstm(x)
        last_hidden = hn[-1][0]
        last_hidden = self.output_batchnorm(last_hidden)
        last_hidden = self.output_dropout(last_hidden)
        out1 = self.linear(last_hidden)

        # Sort it back
        # out1 = out1[x.data.new(unsort_idx).long()]
        return out1

