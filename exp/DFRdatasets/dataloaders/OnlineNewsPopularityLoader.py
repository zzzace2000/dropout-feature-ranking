import numpy as np
import pandas as pd
import torch.nn as nn

from .MLPLoaderBase import MLPLoaderBase


class OnlineNewsPopularityLoader(MLPLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, mode='regression', **kwargs):
        kwargs['mode'] = mode
        super(OnlineNewsPopularityLoader, self).__init__(**kwargs)

    def _load_data(self, testfold=4):
        data = pd.read_csv('data/OnlineNewsPopularity/normalized_new.csv').values
        x = data[:, :-1]
        y = data[:, -1:]
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        alltrainset, testset = self.split_train_and_test((x, y), testfold=testfold)

        # Cut 10% as validation set for NN
        trainset, valset = self.split_valset(alltrainset, ratio=0.1)

        return alltrainset, trainset, valset, testset

    def init_hyperamaters(self):
        return {
            'dimensions': [59, 80, 80, 80, 1],
            'epochs': 100,
            'epoch_print': 1,
            'weight_lr': 1e-3,
            'lookahead': 9,
            'lr_patience': 4,
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
            'epochs': 200,
            'epoch_print': 1,
            'rw_max': 30,
            'loss_criteria': nn.MSELoss(),
        }

    def get_top_indices(self):
        return [1, 2, 3, 5, 10, 20, 35, 59]

    def _get_random_sample_hyperparams(self):
        n_hidden = np.random.randint(60, 150)
        n_layers = np.random.randint(0, 6)

        dimensions = [59, 1]
        for i in range(n_layers):
            dimensions.insert(1, n_hidden)

        return {'dimensions': dimensions}


class ClassificationONPLoader(OnlineNewsPopularityLoader):
    def __init__(self, **kwargs):
        kwargs['mode'] = 'classification'
        super(ClassificationONPLoader, self).__init__(**kwargs)

    def _load_data(self, testfold=4):
        data = pd.read_csv('data/OnlineNewsPopularity/normalized_classification.csv').values
        x = data[:, :-1]
        y = data[:, -1]
        x = x.astype(np.float32)
        y = y.astype(np.int64)

        alltrainset, testset = self.split_train_and_test((x, y), testfold=testfold)

        # Cut 10% as validation set for NN
        trainset, valset = self.split_valset(alltrainset, ratio=0.1)

        return alltrainset, trainset, valset, testset

    def init_hyperamaters(self):
        return {
            'dimensions': [59, 80, 80, 80, 2],
            'epochs': 100,
            'epoch_print': 1,
            'weight_lr': 1e-3,
            'lookahead': 9,
            'lr_patience': 3,
            'verbose': 1,
            'weight_decay': 1e-5,
            'loss_criteria': nn.CrossEntropyLoss(),
        }

    def init_bdnet_hyperparams(self):
        return {
            'reg_coef': 0.005,
            'ard_init': 0.,
            'lr': 0.002,
            'verbose': 1,
            'epochs': 150,
            'epoch_print': 1,
            'rw_max': 30,
            'loss_criteria': nn.CrossEntropyLoss(),
        }

    def _get_random_sample_hyperparams(self):
        n_hidden = np.random.randint(60, 150)
        n_layers = np.random.randint(0, 6)

        dimensions = [59, 2]
        for i in range(n_layers):
            dimensions.insert(1, n_hidden)

        return {'dimensions': dimensions}