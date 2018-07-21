import numpy as np
import pandas as pd
import torch.nn as nn

from .MLPLoaderBase import MLPLoaderBase


class WineQaulityLoader(MLPLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, **kwargs):
        kwargs['mode'] = 'regression'
        super(WineQaulityLoader, self).__init__(**kwargs)

    def init_hyperamaters(self):
        return {
            'dimensions': [11, 86, 86, 1],
            'epochs': 600,
            'epoch_print': 10,
            'weight_lr': 1e-3,
            'lookahead': 20,
            'lr_patience': 9,
            'verbose': 1,
            'weight_decay': 0.,
            'loss_criteria': nn.MSELoss(),
        }

    def init_bdnet_hyperparams(self):
        return {
            'reg_coef': 0.01,
            'ard_init': 0.,
            'lr': 0.001,
            'verbose': 1,
            'epochs': 400,
            'epoch_print': 10,
            'loss_criteria': nn.MSELoss(),
        }

    def _load_data(self, testfold=4):
        x = pd.read_csv('data/wineqaulity/white_x.csv').values
        y = pd.read_csv('data/wineqaulity/white_y.csv').values
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        alltrainset, testset = self.split_train_and_test((x, y), testfold=testfold)

        # Cut 10% as validation set for NN
        trainset, valset = self.split_valset(alltrainset, ratio=0.1)

        return alltrainset, trainset, valset, testset

    def get_top_indices(self):
        return [1, 2, 3, 5, 8, 11]

    def _get_random_sample_hyperparams(self):
        n_hidden = np.random.randint(10, 100)
        n_layers = np.random.randint(0, 5)

        dimensions = [11, 1]
        for i in range(n_layers):
            dimensions.insert(1, n_hidden)

        return {'dimensions': dimensions}
