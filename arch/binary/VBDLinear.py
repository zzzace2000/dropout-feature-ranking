import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from .StochasticBinaryNeuron import sb_neuron
import numpy as np
import torch.optim as optim
import time
from .VBDBase import VBDBase
import torch.optim as optim


class VBDClassification(VBDBase):
    def __init__(self, dim_input, dim_output, prior_p, thresh=0, ard_init=1,
                 anneal=1.05, anneal_max=100, rw_max=20, name=None, reg_coef=1.):
        super(VBDClassification, self).__init__(dim_input, dim_output, thresh, ard_init,
                                                anneal, anneal_max, rw_max, name)
        self.prior_p = prior_p
        self.reg_coef = reg_coef

    def eval_criteria(self, outputs, targets):
        targets = targets.long()
        pred_val, pred_pos = outputs.data.max(dim=1)
        acc = (pred_pos == targets.data).sum()
        return -acc

    def sgvloss(self, outputs, targets, rw, num_samples):
        pred_loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        reg_loss = rw * self.reg_coef * self.eval_reg() / num_samples

        return pred_loss + reg_loss, pred_loss, reg_loss

    def eval_reg(self):
        # Return KL term for the Bernoulli
        p = torch.sigmoid(self.logit_p)

        KL = -p * (torch.log(p) - np.log(self.prior_p)) \
             - (1 - p) * (torch.log(1 - p) - np.log(1 - self.prior_p))

        return -KL.sum()


class VBDLinear(VBDClassification):
    def __init__(self, dim_input, dim_output, prior_p, thresh=0, ard_init=1,
                 anneal=1.05, anneal_max=100, rw_max=20, name=None, neuron=sb_neuron,
                 reg_coef=1.):
        super(VBDLinear, self).__init__(dim_input, dim_output, prior_p, thresh, ard_init,
                                        anneal, anneal_max, rw_max, name, reg_coef)
        self.W_linear = Parameter(torch.FloatTensor(dim_input, dim_output))
        self.b = Parameter(torch.FloatTensor(dim_output))

        stdv = 1. / math.sqrt(self.dim_input)

        self.W_linear.data.normal_(0, stdv)
        self.b.data.fill_(0)

        self.neuron = neuron

    def forward(self, input, epoch=1, stochastic=False, testing=False, thresh=None, train_clip=False):
        logit_p = self.clip(self.logit_p)
        if thresh is None:
            thresh = self.thresh

        mask = self.neuron(logit_p, stochastic=stochastic, testing=testing,
                           anneal_slope=self.anneal_policy(epoch))
        if train_clip:
            mask.data[logit_p.data < thresh] = 0
        weights = self.W_linear * mask

        out = F.linear(input, weights.t(), self.b)
        return out


class VBDSharedWeight(VBDClassification):
    def __init__(self, dim_input, dim_hidden, dim_output, prior_p, thresh=0, ard_init=1,
                 anneal=1.05, anneal_max=100, rw_max=20, neuron=sb_neuron):
        super(VBDSharedWeight, self).__init__(dim_input, dim_output, prior_p, thresh, ard_init,
                                              anneal, anneal_max, rw_max)

        self.dim_hidden = dim_hidden
        self.W = Parameter(torch.Tensor(dim_input, dim_hidden))
        self.b = Parameter(torch.Tensor(dim_hidden))

        self.V = Parameter(torch.Tensor(dim_hidden, dim_output))
        self.c = Parameter(torch.Tensor(dim_output))

        stdv = 1. / math.sqrt(self.dim_input)

        self.W.data.normal_(0, stdv)
        self.b.data.fill_(0)
        self.V.data.normal_(0, stdv)
        self.c.data.fill_(0)

        self.nonlinearity = F.relu
        self.neuron = neuron

    def forward(self, input, epoch=1, stochastic=False, testing=False, thresh=None, train_clip=False):
        logit_p = self.clip(self.logit_p)
        if thresh is None:
            thresh = self.thresh

        z2_mu = []
        for d in range(self.dim_output):
            mask = self.neuron(logit_p[:, d:(d+1)].t(), testing=testing, stochastic=stochastic,
                               anneal_slope=self.anneal_policy(epoch))
            if train_clip:
                mask.data[logit_p.data[:, d:(d+1)] < thresh] = 0

            masked_input = input * mask.expand_as(input)

            z_d = F.linear(masked_input, self.W.t(), self.b)
            h_d = self.nonlinearity(z_d)

            z2_d = torch.mm(h_d, self.V[:, d:(d+1)])
            z2_mu.append(z2_d)

        mu = torch.cat(z2_mu, 1)
        mu = mu + self.c.expand_as(mu)

        return mu


# VBDLinear but with binary error!
class VBDLinearBCE(VBDLinear):
    def sgvloss(self, outputs, targets, rw, num_samples):
        targets = targets.float()
        pred_loss = torch.nn.BCELoss(size_average=True)(outputs, targets)
        reg_loss = rw * self.reg_coef * self.eval_reg() / num_samples

        return pred_loss + reg_loss, pred_loss, reg_loss

    def eval_criteria(self, outputs, targets):
        targets = targets.int()
        pred_val = (outputs > 0.5).int()
        acc = (pred_val.data == targets.data).sum()
        return -acc

    def forward(self, input, epoch=1, stochastic=False, testing=False, thresh=None, train_clip=False):
        val = super(VBDLinearBCE, self).forward(input, epoch, stochastic, testing,
                                                thresh, train_clip)
        return F.sigmoid(val)