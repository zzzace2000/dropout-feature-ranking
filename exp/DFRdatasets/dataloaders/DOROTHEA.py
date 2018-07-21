import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from arch.sensitivity.DFR_BDNet import L1BDNet
from exp.DFRdatasets.models.MLP import MLP, ClassificationMLP
from .MLPLoaderBase import MLPLoaderBase


class DOROTHEA(MLPLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, mode='classification', **kwargs):
        kwargs['mode'] = mode
        super(DOROTHEA, self).__init__(**kwargs)

    def init_hyperamaters(self):
        return {
            'dimensions': [100000, 2],
            'epochs': 300,
            'epoch_print': 1,
            'weight_lr': 1e-3,
            'lookahead': 20,
            'lr_patience': 8,
            'verbose': 1,
            'weight_decay': 0.,
            'loss_criteria': nn.CrossEntropyLoss(),
        }

    def init_bdnet_hyperparams(self):
        return {
            'reg_coef': 0.01,
            'ard_init': 0.,
            'lr': 0.005,
            'weights_lr': 0.001,
            'verbose': 1,
            'epochs': 120,
            'epoch_print': 5,
            'rw_max': 20,
            'loss_criteria': nn.CrossEntropyLoss(),
            # 'annealing': 200,
        }

    def load_x_and_y(self):
        with open('data/DOROTHEA/DOROTHEA/dorothea_train.data') as fp:
            train = fp.readlines()
        with open('data/DOROTHEA/DOROTHEA/dorothea_valid.data') as fp:
            valid = fp.readlines()

        all_data = train + valid
        data_x = np.zeros((len(all_data), 100000))

        for line_idx, line in enumerate(all_data):
            line = line.strip().split(' ')

            for t in line:
                idx = int(t) - 1
                if idx < 0:
                    print(idx)
                data_x[line_idx, idx] = 1

        y_train = pd.read_csv('data/DOROTHEA/DOROTHEA/dorothea_train.labels',
                              sep=' ', header=None, index_col=None)
        y_valid = pd.read_csv('data/DOROTHEA/DOROTHEA/dorothea_valid.labels',
                              sep=' ', header=None, index_col=None)

        all_y = pd.concat([y_train, y_valid], axis=0, ignore_index=True)
        all_y[all_y == -1] = 0

        data_x = data_x.astype(np.float32)
        all_y = all_y.astype(np.int64).values.ravel()
        return data_x, all_y

    def _load_data(self, testfold=4):
        x, y = self.load_x_and_y()
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

