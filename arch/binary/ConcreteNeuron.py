import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

EPSILON = np.finfo(float).eps

def concrete_neuron(logit_p, testing=False, temp=1.0 / 10.0, **kwargs):
    '''
    Use concrete distribution to approximate binary output. Here input is logit(keep_prob).
    '''
    if testing:
        result = logit_p.data.new().resize_as_(logit_p.data).fill_(1.)
        result[logit_p.data < 0.] = 0.
        return Variable(result)

    # Note that p is the retain probability here
    p = torch.sigmoid(logit_p)
    unif_noise = Variable(logit_p.data.new().resize_as_(logit_p.data).uniform_())

    approx = (
        torch.log(1. - p + EPSILON)
        - torch.log(p + EPSILON)
        + torch.log(unif_noise + EPSILON)
        - torch.log(1. - unif_noise + EPSILON)
    )
    drop_prob = torch.sigmoid(approx / temp)
    return (1. - drop_prob)

def concrete_dropout_neuron(dropout_p, temp=1.0 / 10.0, **kwargs):
    '''
    Use concrete distribution to approximate binary output. Here input is logit(dropout_prob).
    '''
    # Note that p is the dropout probability here
    unif_noise = Variable(dropout_p.data.new().resize_as_(dropout_p.data).uniform_())

    approx = (
        torch.log(dropout_p + EPSILON)
        - torch.log(1. - dropout_p + EPSILON)
        + torch.log(unif_noise + EPSILON)
        - torch.log(1. - unif_noise + EPSILON)
    )
    approx_output = torch.sigmoid(approx / temp)
    return 1 - approx_output

def multiclass_concrete_neuron(log_alpha, temp=0.1, **kwargs):
    '''
    Use concrete distribution to approximate multiclass output.
    :param log_alpha: np array [N, nclass]
    :return: Sample value: np array [N, nclass]
    '''
    # Note that p is the dropout probability here
    alpha = torch.exp(log_alpha)
    uniform = Variable(log_alpha.data.new().resize_as_(log_alpha.data).uniform_())
    gumbel = - torch.log(- torch.log(uniform + EPSILON) + EPSILON)
    logit = (torch.log(alpha + EPSILON) + gumbel) / temp

    return F.softmax(logit)

if __name__ == '__main__':

    def test_val(p_val):
        p_tensor = p_val * torch.ones(1)
        logit_p = Variable(torch.log(p_tensor) - torch.log(1 - p_tensor))
        arr = [concrete_neuron(logit_p).data[0] for i in range(100)]
        print('retain prob:', p_val, 'Average over 100:', np.mean(arr))
    # Keep probability is 0.5
    test_val(0.5)
    test_val(0.1)
    test_val(0.9)

    def test_val(p_val):
        p_tensor = p_val * torch.ones(1)
        logit_p = Variable(torch.log(p_tensor) - torch.log(1 - p_tensor))
        arr = [concrete_dropout_neuron(logit_p).data[0] for i in range(100)]
        print('dropout prob:', p_val, 'Average over 100:', np.mean(arr))
        print(arr[0])

    test_val(0.5)
    test_val(0.1)
    test_val(0.9)

    log_alpha = Variable(torch.ones(1, 4))
    print(multiclass_concrete_neuron(log_alpha))

    log_alpha[0, :2] = 5
    print(multiclass_concrete_neuron(log_alpha))
