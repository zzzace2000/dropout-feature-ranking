from torch.autograd import Function
import torch
from torch.autograd import Variable

class StochasticBinaryNeuron(Function):
    def __init__(self, anneal_slope=1.):
        super(StochasticBinaryNeuron, self).__init__()
        self.anneal_slope = anneal_slope

    def forward(self, input):
        self.input = input
        p = hard_sigmoid(input)
        return torch.bernoulli(p)

    # def backward(self, grad_output):
    #     grad_input = grad_output * self.anneal_slope / 2.
    #     criteria = (self.input < (-1. / self.anneal_slope)) | (self.input > (1. / self.anneal_slope))
    #     grad_input[criteria] = 0.
    #     return grad_input

    # Soft Sigmoid backward function
    def backward(self, grad_output):
        p = torch.sigmoid(self.input * self.anneal_slope)
        grad_input = grad_output * p * (1 - p) * self.anneal_slope

        return grad_input

class DeterministicBinaryNeuron(StochasticBinaryNeuron):
    def forward(self, input):
        self.input = input
        output = input.new().resize_as_(input).zero_()
        output[input > 0] = 1
        return output

class SoftNeuron(StochasticBinaryNeuron):
    def forward(self, input):
        self.input = input
        return hard_sigmoid(input, anneal=self.anneal_slope)

def sb_neuron(input, stochastic=False, testing=False, thresh=0, anneal_slope=1.):
    if stochastic:
        if not testing:
            return StochasticBinaryNeuron(anneal_slope)(input)

        mask = input.data.new().resize_as_(input).fill_(1.)
        tmp = (input.data < thresh)
        mask.masked_fill_(tmp, 0.)
        return Variable(mask)

    return DeterministicBinaryNeuron(anneal_slope)(input)

def soft_neuron(input, anneal_slope=1., **kwargs):
    return SoftNeuron(anneal_slope)(input)

def hard_sigmoid(x, anneal=1.):
    x = (anneal * x + 1) / 2
    return x.clamp(0, 1)

if __name__ == '__main__':
    print('asd')

    a = Variable(torch.ones(2, 2), requires_grad=False)

    gate = Variable(-10 * torch.ones(2, 2), requires_grad=True)

    neuron = sb_neuron(gate, stochastic=True, anneal_slope=1.)

    result = (a * neuron).sum()
    result.backward()

    print(neuron)
    print(result)
    print((gate.grad))


