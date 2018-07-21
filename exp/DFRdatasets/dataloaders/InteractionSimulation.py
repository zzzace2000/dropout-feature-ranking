import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from arch.sensitivity.DFR_BDNet import L1BDNet
from exp.DFRdatasets.models.MLP import MLP, ClassificationMLP
from .MLPLoaderBase import MLPLoaderBase
from scipy.stats import spearmanr


class InteractionSimulation(MLPLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, mode='regression', **kwargs):
        kwargs['mode'] = mode
        super(InteractionSimulation, self).__init__(**kwargs)
        self.batch_size = 128

        self._init_sim_dataset()
        # Register callback
        def register_callback(vbd_net):
            def metrics_callback():
                rank = -vbd_net.logit_p.data.numpy()
                return 'spearman: %f' % spearmanr(
                    self.gnd_truth_rank, rank).correlation
            vbd_net.register_epoch_callback(metrics_callback)

        self.vbd_net_callback = register_callback

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
        return super(InteractionSimulation, self)._vis_plot_rank(
            rank, title, thewin)

    def init_hyperamaters(self):
        return {
            'dimensions': [40, 40, 20, 1],
            'epochs': 100,
            'epoch_print': 1,
            'weight_lr': 1e-3,
            'lookahead': 10,
            'lr_patience': 3,
            'verbose': 1,
            'weight_decay': 1e-5,
            'loss_criteria': nn.MSELoss(),
        }

    # def _nn_rank(self, nn, testfold=4, bdnet_class=L1BDNet):
    #     super()

    def init_bdnet_hyperparams(self):
        return {
            'reg_coef': 0.05,
            'ard_init': 0.,
            'lr': 0.001,
            'verbose': 1,
            'epochs': 200,
            'epoch_print': 1,
            'rw_max': 30,
            'loss_criteria': nn.MSELoss(),
        }

    def _load_data(self, testfold=4, N=10000):
        x, y = self.x, self.y

        alltrainset, testset = self.split_train_and_test((x, y), testfold=testfold)

        # Cut 10% as validation set for NN
        trainset, valset = self.split_valset(alltrainset, ratio=0.1)

        return alltrainset, trainset, valset, testset

    def get_top_indices(self):
        return [1, 2, 5, 10, 20, 30, 40]

    def _get_random_sample_hyperparams(self):
        n_hidden = np.random.randint(40, 200)
        n_layers = np.random.randint(0, 5)

        dimensions = [40, 2]
        for i in range(n_layers):
            dimensions.insert(1, n_hidden)

        return {'dimensions': dimensions}

# def cal_spearman_corr(vbd_net):
#     return 'spearman: %f' % \
#            spearmanr(self.gnd_truth_rank,
#                      -vbd_net.logit_p.data.numpy()).correlation


class NoInteractionSimulation(InteractionSimulation):
    def _init_sim_dataset(self, N=10000):
        # Produce datasets
        x = np.random.normal(0, 1, size=(N, 40)).astype(np.float32)
        y = np.zeros((N, 1), dtype=np.float32)
        # Interaction term among x
        for i in range(0, 20):
            bern_noise = np.random.binomial(1, 1. - 0.05 * i, size=(N,))
            y[:, 0] = y[:, 0] + x[:, i] * bern_noise

        self.x, self.y = x, y
        self.gnd_truth_rank = np.zeros((40,))
        for i in range(0, 20):
            self.gnd_truth_rank[i] = (20 - i)


class CorrelatedNoInteractionSimulation(InteractionSimulation):
    def _init_sim_dataset(self, N=10000):
        # Produce datasets
        corr_val = 0.5
        mean_mat = np.zeros((40,))
        corr_mat = (1. - corr_val) * np.eye(40) + corr_val
        x = np.random.multivariate_normal(mean_mat, corr_mat, size=N)
        x = x.astype(np.float32)
        y = np.zeros((N, 1), dtype=np.float32)
        # Interaction term among x
        for i in range(0, 20):
            bern_noise = np.random.binomial(1, 1. - 0.05 * i, size=(N,))
            y[:, 0] = y[:, 0] + x[:, i] * bern_noise

        self.x, self.y = x, y
        self.gnd_truth_rank = np.zeros((40,))
        for i in range(0, 20):
            self.gnd_truth_rank[i] = (20 - i)


class CorrelatedInteractionSimulation(InteractionSimulation):
    def _init_sim_dataset(self, N=10000):
        # Produce datasets
        corr_val = 0.5
        mean_mat = np.zeros((40,))
        corr_mat = (1. - corr_val) * np.eye(40) + corr_val
        x = np.random.multivariate_normal(mean_mat, corr_mat, size=N)
        x = x.astype(np.float32)
        y = np.zeros((N, 1), dtype=np.float32)
        # Interaction term among x
        for i in range(0, 20, 2):
            bern_noise = np.random.binomial(1, 1. - 0.05 * i, size=(N,))
            y[:, 0] = y[:, 0] + x[:, i] * x[:, (i + 1)] * bern_noise

        self.x, self.y = x, y
        self.gnd_truth_rank = np.zeros((40,))
        for i in range(0, 20, 2):
            self.gnd_truth_rank[i:(i + 2)] = (20 - i)

        print('y mean, var:', y.mean(), y.var())


class MoreInteractionSimulation(InteractionSimulation):
    def __init__(self, **kwargs):
        super(MoreInteractionSimulation, self).__init__(**kwargs)

    def _init_sim_dataset(self, N=10000):
        # Produce datasets
        x = np.random.normal(0, 1, size=(N, 60)).astype(np.float32)
        y = np.zeros((N, 1), dtype=np.float32)
        # Interaction term among x
        for i in range(0, 30, 3):
            bern_noise = np.random.binomial(1, 1. - 0.03 * i, size=(N,))
            y[:, 0] = y[:, 0] + (x[:, i] * x[:, (i + 1)] + x[:, (i + 2)]) * bern_noise

        self.x, self.y = x, y
        self.gnd_truth_rank = np.zeros((60,))
        for i in range(0, 30, 3):
            self.gnd_truth_rank[i:(i + 3)] = (30 - i)

        print('y mean, var:', y.mean(), y.var())

    def init_hyperamaters(self):
        return {
            'dimensions': [60, 80, 80, 40, 20, 1],
            'epochs': 300,
            'epoch_print': 1,
            'weight_lr': 1e-3,
            'lookahead': 13,
            'lr_patience': 4,
            'verbose': 1,
            'weight_decay': 1e-5,
            'loss_criteria': nn.MSELoss(),
        }