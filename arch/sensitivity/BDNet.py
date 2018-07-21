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
from torch.distributions import Bernoulli


def logit(x):
    return np.log(x) - np.log(1. - x)


class BDNet(DNetBase):
    def __init__(self, dropout_param_size, trained_classifier,
                 loss_criteria=nn.CrossEntropyLoss(),
                 ard_init=-5, lr=0.01, reg_coef=1., rw_max=30, cuda_enabled=False,
                 verbose=0, estop_num=None, clip_max=100, flip_val=0.,
                 flip_train=False, **kwargs):
        super(BDNet, self).__init__(trained_classifier, loss_criteria,
                                    lr, reg_coef, rw_max, cuda_enabled,
                                    verbose, estop_num, clip_max=clip_max, **kwargs)
        self.dropout_param_size = dropout_param_size
        self.ard_init = ard_init

        self.logit_p = Parameter(torch.Tensor(*dropout_param_size))

        # self.flip_param = Parameter(torch.Tensor(*dropout_param_size).fill_(flip_val))

        self.flip_val = flip_val
        self.flip_train = flip_train

        self.evaluate = False

    def param_name(self):
        return 'logit_p'

    def my_parameters(self):
        arr = [self.logit_p]
        # if self.flip_train:
        #     arr += [self.flip_param]
        return arr

    def initialize(self, train_loader, epochs):
        super(BDNet, self).initialize(train_loader, epochs)
        self.get_param().data.fill_(self.ard_init)
        # self.flip_param.data.fill_(self.flip_val)

    def sampled_noisy_input(self, input):
        # logit_p = self.clip(self.logit_p)

        if self.evaluate:
            expanded_logit_p = self.logit_p.unsqueeze(0).expand(
                input.size(0), *self.logit_p.size())
            p = torch.sigmoid(expanded_logit_p)

            scaled_input = input * (1. - p)
            return scaled_input

        bern_val = self.sampled_from_logit_p(input.size(0))

        # expanded_flip_param = self.flip_param.unsqueeze(0).expand_as(input)
        noised_input = input * bern_val

        return noised_input

    def sampled_from_logit_p(self, num_samples):
        expanded_logit_p = self.logit_p.unsqueeze(0).expand(num_samples, *self.logit_p.size())
        dropout_p = torch.sigmoid(expanded_logit_p)
        bern_val = concrete_dropout_neuron(dropout_p)
        return bern_val

    def set_eval(self, is_eval=True):
        self.evaluate = is_eval

    def eval_loader_with_loss_and_acc(self, loader, cuda=False):
        self.set_eval(True)
        result = super(BDNet, self).eval_loader_with_loss_and_acc(loader, cuda)
        self.set_eval(False)
        return result

    def forward(self, input):
        noised_input = self.sampled_noisy_input(input)

        if self.verbose > 1:
            print('mean_logit_p: %f, Noise mean: %f, Noise var: %f' % (
                self.logit_p.data.mean(), (noised_input - input).data.mean(),
                (noised_input - input).data.var()))

        return self.trained_classifier(noised_input)

    def eval_reg(self):
        dropout_p = torch.sigmoid(self.logit_p)
        loss = (1. - dropout_p)
        return loss

    def get_importance_vector(self):
        # Return dropout rate
        imp_vector = torch.sigmoid(-self.logit_p.data[0, ...])
        imp_vector -= 0.5

        return imp_vector

L1BDNet = BDNet

class SigmoidBDNet(BDNet):
    def __init__(self, num_features, annealing=200, **kwargs):
        super(SigmoidBDNet, self).__init__(**kwargs)
        self.num_features = num_features
        self.annealing = annealing

    def eval_reg(self):
        # Return a vector of sigmoid loss lol!
        diff_feature_num = self.bern_val.sum(dim=1) - self.num_features
        return F.relu(1. * diff_feature_num)


class MiddleBDNet(BDNet):
    """ Penalize the dropout rate toward 0.5. The lambda needs to times D """
    def eval_reg(self):
        p = torch.sigmoid(self.logit_p)
        loss = F.relu(p - 0.5) + F.relu(0.5 - p)
        return loss * self.logit_p.shape[-1]


class TrainWeightsL1BDNet(BDNet):
    def __init__(self, weights_lr, **kwargs):
        super(TrainWeightsL1BDNet, self).__init__(**kwargs)
        self.weights_lr = weights_lr

        # Preserve the gradient in pretrained model
        self.trained_classifier.train()
        for param in self.trained_classifier.parameters():
            param.requires_grad = True

    def my_parameters(self):
        dropout_params = super(TrainWeightsL1BDNet, self).my_parameters()

        return [{'params': dropout_params},
                {'params': self.trained_classifier.parameters(),
                 'lr': self.weights_lr}]


class REINFORCE_L1BDNet(BDNet):
    '''
    Optimize dropout rate by REINFORCE estimator
    '''
    def __init__(self, norm_reward=False, **kwargs):
        # Replace the passed in loss criteria as the defined reinforce loss!
        kwargs['loss_criteria'] = self.reinforce_loss
        super(REINFORCE_L1BDNet, self).__init__(**kwargs)

        self.log_prob = None
        self.bern_val = None
        self.norm_reward = norm_reward

    def sampled_from_logit_p(self, num_samples):
        expanded_logit_p = self.logit_p.unsqueeze(0).expand(num_samples, *self.logit_p.size())

        # Note that p is the dropout probability here
        drop_p = torch.sigmoid(expanded_logit_p)
        m = Bernoulli(1. - drop_p)

        bern_val = m.sample()
        if self.log_prob is not None:
            raise Exception('Log probability should be cleaned up after use')
        self.log_prob = m.log_prob(bern_val)

        self.bern_val = bern_val
        return bern_val

    def reinforce_loss(self, outputs, targets):
        '''
        Supports multi-class loss for reinforce estimator
        :param outputs: B x C tensors
        :param targets: B Long Tensor
        '''
        # if evaluation, just returns the normal cross entropy
        if self.evaluate:
            return nn.CrossEntropyLoss()(outputs, targets)

        rewards = torch.gather(F.log_softmax(outputs, dim=1), dim=1, index=targets.unsqueeze(-1))

        # Stablize training by normalizing reward
        if self.norm_reward:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        reinforce_loss = -(self.log_prob * rewards)
        reinforce_loss = reinforce_loss.sum() * 1.0 / reinforce_loss.size(0)

        return reinforce_loss

    def eval_reg(self):
        # if evaluation, just returns the normal cross entropy
        if self.evaluate:
            return self.bern_val * 1.0 / self.bern_val.size(0)

        rewards = self.bern_val.sum(dim=1, keepdim=True)
        if self.norm_reward:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Return a vector of loss. Normalized by batch size M
        reg_loss = -self.log_prob * rewards
        self.log_prob = None
        return reg_loss * 1.0 / reg_loss.size(0)


if __name__ == '__main__':
    def plot_bd_loss(prior_p=0.999, filename='bd_loss.png'):
        # Plot the reg cost curve
        p = torch.arange(0.001, 0.999, step=0.001)
        KL = p * (torch.log(p) - np.log(prior_p)) \
             + (1 - p) * (torch.log(1 - p) - np.log(1 - prior_p))

        x = p.numpy()
        y = KL.numpy()

        import matplotlib.pyplot as plt

        plt.plot(x, y)
        plt.ylabel('reg loss')
        plt.xlabel('dropout prob p')
        plt.title('Reg loss')
        plt.show()
        # plt.savefig(filename, dpi=300)
        plt.close()

    # plot_bd_loss()
    # plot_bd_loss(prior_p=0.5, filename='bd_loss_0.5.png')

    p = np.array([0.45, 0.55])
    KL = p * (np.log(p) - np.log(0.5)) \
         + (1 - p) * (np.log(1 - p) - np.log(1 - 0.5))
    print(KL)