import numpy as np
import pandas as pd
import torch.nn as nn

from .MLPLoaderBase import MLPLoaderBase
from sklearn.utils import shuffle


class HIGGS(MLPLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, mode='classification', **kwargs):
        kwargs['mode'] = mode
        super(HIGGS, self).__init__(**kwargs)
        self.batch_size = 512

    def init_hyperamaters(self):
        return {
            'dimensions': [21, 200, 100, 50, 25, 2],
            'epochs': 20,
            'epoch_print': 1,
            'weight_lr': 1e-4,
            'lookahead': 3,
            'lr_patience': 1,
            'verbose': 1,
            'weight_decay': 1e-4,
            'loss_criteria': nn.CrossEntropyLoss(),
        }

    def init_bdnet_hyperparams(self):
        return {
            'reg_coef': 0.001,
            'ard_init': 0.,
            'lr': 0.0005,
            'verbose': 1,
            'epochs': 5,
            'epoch_print': 1,
            'rw_max': 1,
            'loss_criteria': nn.CrossEntropyLoss(),
        }

    def _load_data(self, testfold=4):
        x = pd.read_csv('data/HIGGS/x.csv').values
        y = pd.read_csv('data/HIGGS/y.csv', header=None).values.ravel()
        x = x.astype(np.float32)
        y = y.astype(np.int64)

        alltrainset, testset = self.split_train_and_test((x, y), testfold=testfold)

        # Cut 10% as validation set for NN
        trainset, valset = self.split_valset(alltrainset, ratio=0.1)

        return alltrainset, trainset, valset, testset

    def get_top_indices(self):
        return [1, 2, 3, 5, 10, 15, 21]

    def _get_random_sample_hyperparams(self):
        return {'dimensions': self.hyper_params['dimensions']}
        # n_hidden = np.random.randint(50, 200)
        # n_layers = np.random.randint(0, 5)
        #
        # dimensions = [50, 2]
        # for i in range(n_layers):
        #     dimensions.insert(1, n_hidden)
        #
        # return {'dimensions': dimensions}
