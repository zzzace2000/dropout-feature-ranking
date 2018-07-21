import numpy as np
import pandas as pd
import torch.nn as nn

from .MLPLoaderBase import MLPLoaderBase


class YearMSD(MLPLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, mode='regression', **kwargs):
        kwargs['mode'] = mode
        super(YearMSD, self).__init__(**kwargs)

    def _load_data(self, testfold=4):
        data = pd.read_csv('data/yearpredictionMSD/normalized.csv').values
        x = data[:, 1:]
        y = data[:, 0:1]
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        alltrainset, testset = self.split_train_and_test((x, y), testfold=testfold)

        # Cut 10% as validation set for NN
        trainset, valset = self.split_valset(alltrainset, ratio=0.1)

        return alltrainset, trainset, valset, testset

    def init_hyperamaters(self):
        return {
            'dimensions': [90, 200, 100, 80, 40, 20, 1],
            'epochs': 100,
            'epoch_print': 1,
            'weight_lr': 5e-4,
            'lookahead': 5,
            'lr_patience': 2,
            'verbose': 1,
            'weight_decay': 1e-5,
            'loss_criteria': nn.MSELoss(),
        }

    def init_bdnet_hyperparams(self):
        return {
            'reg_coef': 0.01,
            'ard_init': 0.,
            'lr': 0.002,
            'verbose': 1,
            'epochs': 100,
            'epoch_print': 1,
            'rw_max': 20,
            'loss_criteria': nn.MSELoss(),
        }

    def get_top_indices(self):
        return [1, 2, 3, 5, 10, 20, 40, 60, 90]

    def _get_random_sample_hyperparams(self):
        n_hidden = np.random.randint(100, 200)
        n_layers = np.random.randint(0, 7)

        dimensions = [90, 1]
        for i in range(n_layers):
            dimensions.insert(1, n_hidden)

        return {'dimensions': dimensions}
