import torch
from torch.autograd import Variable


class VarDropoutWrapper:
    def __init__(self):
        print('Should not initiate this class directly!')
        exit(-1)

    @staticmethod
    def clip(mtx, to=8):
        mtx.data[mtx.data > to] = to
        mtx.data[mtx.data < -to] = -to
        return mtx

    def L2_error(self):
        Mseloss = torch.nn.MSELoss(size_average=False)

        l2loss = 0.
        for name, param in self.named_parameters():
            if name.startswith(('W', 'V', 'b', 'c')):
                zero_tensor = Variable(torch.zeros(*param.size()))
                if self.cuda_enabled:
                    zero_tensor = zero_tensor.cuda()
                l2loss += Mseloss(param, zero_tensor)
        return l2loss

    def get_sparsity(self, **kwargs):
        return '%.4f with threshold %.1f' % ((self.log_alpha.data > self.thresh).sum() * 1.0
                                        / torch.numel(self.log_alpha.data), self.thresh)

    def get_alpha_range(self):
        mask = torch.ones(*self.log_alpha.size())
        for i in range(self.input_size):
            mask[i, i] = 0
        mask = Variable(mask == 1)

        log_alpha = self.log_alpha.masked_select(mask)
        return '%.2f, %.2f' % (log_alpha.data.min(), log_alpha.data.max())