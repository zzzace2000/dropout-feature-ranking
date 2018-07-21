import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from arch.sensitivity.DFR_BDNet import L1BDNet
from exp.DFRdatasets.models.MLP import MLP, ClassificationMLP
from .MLPLoaderBase import MLPLoaderBase


class MADELON(MLPLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, mode='classification', **kwargs):
        kwargs['mode'] = mode
        super(MADELON, self).__init__(**kwargs)

    def init_hyperamaters(self):
        return {
            'dimensions': [500, 2],
            'epochs': 500,
            'epoch_print': 1,
            'weight_lr': 1e-3,
            'lookahead': 20,
            'lr_patience': 10,
            'verbose': 1,
            'weight_decay': 1e-6,
            'loss_criteria': nn.CrossEntropyLoss(),
        }

    def init_bdnet_hyperparams(self):
        return {
            'reg_coef': 0.001,
            'ard_init': -3.,
            'lr': 0.005,
            'weights_lr': 0.001,
            'verbose': 1,
            'epochs': 300,
            'epoch_print': 5,
            'rw_max': 20,
            'loss_criteria': nn.CrossEntropyLoss(),
            # 'annealing': 200,
        }

    def _load_data(self, testfold=4):
        x = pd.read_csv('data/MADELON/x.csv').values
        y = pd.read_csv('data/MADELON/y.csv').values.ravel()
        x = x.astype(np.float32)
        y = y.astype(np.int64)
        print(x.shape, y.shape)

        alltrainset, testset = self.split_train_and_test((x, y), testfold=testfold)

        # Cut 10% as validation set for NN
        trainset, valset = self.split_valset(alltrainset, ratio=0.1)

        return alltrainset, trainset, valset, testset

    def get_top_indices(self):
        return [25, 50, 100, 200, 300, 500, 700, 800, 900, 1000]

    def _get_random_sample_hyperparams(self):
        n_hidden = np.random.randint(39, 200)
        n_layers = np.random.randint(0, 5)

        dimensions = [39, 2]
        for i in range(n_layers):
            dimensions.insert(1, n_hidden)

        return {'dimensions': dimensions}

