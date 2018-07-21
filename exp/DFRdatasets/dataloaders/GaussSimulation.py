import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from arch.sensitivity.BDNet import L1BDNet, TrainWeightsL1BDNet
from exp.DFRdatasets.models.MLP import MLP, ClassificationMLP
from .MLPLoaderBase import MLPLoaderBase
from scipy.stats import spearmanr
from .InteractionSimulation import InteractionSimulation


class GaussSimulation(MLPLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, mode='classification', **kwargs):
        kwargs['mode'] = mode
        super(GaussSimulation, self).__init__(**kwargs)
        self.batch_size = 128

        self._init_sim_dataset()
        # Register callback
        def register_callback(vbd_net):
            def metrics_callback():
                rank = -vbd_net.logit_p.data.cpu().numpy()
                return 'spearman: %f' % spearmanr(
                    self.gnd_truth_rank, rank).correlation
            vbd_net.register_epoch_callback(metrics_callback)

        self.vbd_net_callback = register_callback
        self.nn_cache = False

    def vbd_linear_rank(self, testfold):
        ''' Test vbd net dropout rate in linear layer '''
        # Put in an untrained linear layer
        linearNN = self.nn_class(dimensions=[100, 2],
                                 loss_criteria=nn.CrossEntropyLoss())
        # Rank nn feature based on var dropout
        vbd_net = self._nn_rank(linearNN, testfold, TrainWeightsL1BDNet)

        nn_rank = -vbd_net.logit_p.data.cpu().numpy()
        _, _, test_loader = self._load_pytorch_loader(testfold=testfold)
        _, test_acc = vbd_net.eval_loader_with_loss_and_acc(test_loader)

        return {'rank': nn_rank, 'test_acc': test_acc}

    def _init_sim_dataset(self, N=10000):
        # Produce datasets
        x = np.random.normal(0, 1, size=(N, 40)).astype(np.float32)
        y = np.zeros((N, 1), dtype=np.float32)
        # Interaction term among x
        for i in range(0, 20, 2):
            bern_noise = np.random.binomial(1, 1. - 0.05 * i, size=(N,))
            y[:, 0] = y[:, 0] + x[:, i] * x[:, (i + 1)] * bern_noise

        self.x, self.y = x, y
        self.gnd_truth_rank = np.zeros((40,))
        for i in range(0, 20, 2):
            self.gnd_truth_rank[i:(i + 2)] = (20 - i)

    def _vis_plot_rank(self, rank, title, thewin=None):
        title += '(%.3f)' % spearmanr(self.gnd_truth_rank,
                                      rank).correlation
        return super(GaussSimulation, self)._vis_plot_rank(
            rank, title, thewin)

    def init_hyperamaters(self):
        return {
            'dimensions': [100, 50, 20, 2],
            'epochs': 100,
            'epoch_print': 1,
            'weight_lr': 1e-3,
            'lookahead': 10,
            'lr_patience': 3,
            'verbose': 1,
            'weight_decay': 1e-5,
            'loss_criteria': nn.CrossEntropyLoss(),
        }

    def init_bdnet_hyperparams(self):
        return {
            'reg_coef': 1.,
            'ard_init': 0.,
            'lr': 0.001,
            'weights_lr': 0.002,
            'verbose': 1,
            'epochs': 1000,
            'epoch_print': 20,
            'rw_max': 100,
            'loss_criteria': nn.CrossEntropyLoss(),
        }

    def _load_data(self, testfold=4):
        tmp = testfold.split('_')
        datasize = int(tmp[0])
        corr_val = float(tmp[1])

        alltrainset = self._gen_simulate_data(N=datasize, corr_val=corr_val)
        testset = self._gen_simulate_data(N=3000, corr_val=corr_val)

        # Cut 10% as validation set for NN
        trainset, valset = self.split_valset(alltrainset, ratio=0.1)

        return alltrainset, trainset, valset, testset

    def _gen_simulate_data(self, N, P=100, n_informative=40, noise_var=1., corr_val=0.):
        # Produce datasets
        x = np.zeros((N, P)).astype('float32')
        y = np.random.randint(2, size=N)

        for i in range(N):
            mean_mat = np.array([0.01 * (f + 1) * 2 * (y[i] - 0.5)
                                 for f in range(n_informative)])
            corr_mat = (1. - corr_val) * np.eye(n_informative) + corr_val
            x[i, :n_informative] = np.random.multivariate_normal(mean_mat, corr_mat)

            for feature_idx in range(n_informative, P):
                x[i, feature_idx] = np.random.normal(0, noise_var)

        # Gnd Truth Rank
        self.gnd_truth_rank = np.array(list(range(P)))
        self.gnd_truth_rank[n_informative:] = -1

        x = x.astype(np.float32)
        y = y.astype(np.int64)

        x = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
        return x, y

    def evaluate(self, rank):
        spearman = spearmanr(self.gnd_truth_rank, rank).correlation
        return {'spearman': spearman}


    # def get_top_indices(self):
    #     return [1, 2, 5, 10, 20, 30, 40]

    # def _get_random_sample_hyperparams(self):
    #