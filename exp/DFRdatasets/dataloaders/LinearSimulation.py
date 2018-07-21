import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from arch.sensitivity.DFR_BDNet import L1BDNet
from exp.DFRdatasets.models.MLP import MLP, ClassificationMLP
from .MLPLoaderBase import MLPLoaderBase


class LinearSimulation(MLPLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, mode='regression', n_informative=50, N=100, P=100,
                 noise_var=0.1, weight_magnitude=1, **kwargs):
        kwargs['mode'] = mode
        super(LinearSimulation, self).__init__(**kwargs)

        self.n_informative = n_informative
        self.N = N
        self.P = P
        self.noise_var = noise_var
        self.weight_magnitude = weight_magnitude

    def init_hyperamaters(self):
        return {
            'dimensions': [100, 1],
            'epochs': 500,
            'epoch_print': 1,
            'weight_lr': 1e-3,
            'lookahead': 25,
            'lr_patience': 10,
            'verbose': 1,
            'weight_decay': 0.,
            'loss_criteria': nn.MSELoss(),
        }

    def init_bdnet_hyperparams(self):
        return {
            'reg_coef': 0.1,
            'ard_init': 0.,
            'lr': 0.05,
            'weights_lr': 0.001,
            'verbose': 1,
            'epochs': 1000,
            'epoch_print': 5,
            'rw_max': 30,
            'loss_criteria': nn.MSELoss(),
            # 'annealing': 200,
        }

    def _load_data(self, testfold=4):
        # Produce datasets
        x = np.random.normal(size=(self.N, self.P))

        self.gnd_truth_weight = self.weight_magnitude * np.random.uniform(size=(self.P, 1))
        self.gnd_truth_weight[self.n_informative:, 0] = 0

        y = np.dot(x, self.gnd_truth_weight) + np.random.normal(scale=self.noise_var, size=(self.N, 1))

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        print(x.shape, y.shape)

        alltrainset, testset = self.split_train_and_test((x, y), testfold=testfold)

        # Cut 10% as validation set for NN
        trainset, valset = self.split_valset(alltrainset, ratio=0.1)

        return alltrainset, trainset, valset, testset

    def get_gnd_truth_weight(self):
        return self.gnd_truth_weight

    # def get_top_indices(self):
    #     return [25, 50, 100, 200, 300, 500, 700, 800, 900, 1000]
    #
    # def _get_random_sample_hyperparams(self):
    #     n_hidden = np.random.randint(39, 200)
    #     n_layers = np.random.randint(0, 5)
    #
    #     dimensions = [39, 2]
    #     for i in range(n_layers):
    #         dimensions.insert(1, n_hidden)
    #
    #     return {'dimensions': dimensions}

