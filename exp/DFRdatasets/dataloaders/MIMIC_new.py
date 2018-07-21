import random
import numpy as np

import numpy as np
import pandas as pd
import torch.nn as nn
import copy
from .RNNLoaderBase import RNNLoaderBase
from exp.DFRdatasets.models.LSTM import ClfLSTM
import pickle


class MIMIC_new(RNNLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, **kwargs):
        self.mydata_cache = {}
        self.total_folds = 5
        self.batch_size = 32
        self._my_load_data()

        kwargs['mode'] = 'classification'
        super(MIMIC_new, self).__init__(**kwargs)
        self.nn_class = ClfLSTM
        self.total_folds = 5
        self.batch_size = 32


    def get_time_feature_shape(self):
        return -1, 42

    def load_data(self, testfold=0):
        ''' For sklearn module. So only return train and test set.'''
        alltrainset, _, _, testset = self._load_data(testfold=testfold)

        # reshape as a numpy array.
        def reshape_x(set):
            x, y = set
            x = x.reshape((x.shape[0], -1))
            return x, y

        alltrainset = reshape_x(alltrainset)
        testset = reshape_x(testset)
        return alltrainset, None, None, testset

    def _load_data(self, testfold=4):
        if testfold in self.mydata_cache:
            return self.mydata_cache[testfold]

        # Needs to split by ratio of positive and negative cases
        fold_num = len(self.total_pos_num) // self.total_folds
        pos_idxes_train = np.delete(self.total_pos_num, range(testfold * fold_num, (testfold + 1) * fold_num), axis=0)
        pos_idxes_test = self.total_pos_num[testfold * fold_num:(testfold + 1) * fold_num]

        # Cut 10% as validation!
        val_num = int(pos_idxes_train.shape[0] * 0.9)
        pos_idxes_val = pos_idxes_train[val_num:]
        pos_idxes_train = pos_idxes_train[:val_num]

        fold_num = len(self.total_neg_num) // self.total_folds
        neg_idxes_train = np.delete(self.total_neg_num, range(testfold * fold_num, (testfold + 1) * fold_num), axis=0)
        neg_idxes_test = self.total_neg_num[testfold * fold_num:(testfold + 1) * fold_num]

        # Cut 10% as validation!
        val_num = int(neg_idxes_train.shape[0] * 0.9)
        neg_idxes_val = neg_idxes_train[val_num:]
        neg_idxes_train = neg_idxes_train[:val_num]

        alltrainset = self.all_x[np.concatenate((pos_idxes_train, neg_idxes_train, pos_idxes_val, neg_idxes_val), axis=0)], \
                      self.all_y[np.concatenate((pos_idxes_train, neg_idxes_train, pos_idxes_val, neg_idxes_val), axis=0)]

        trainset = self.all_x[np.concatenate((pos_idxes_train, neg_idxes_train), axis=0)], \
                   self.all_y[np.concatenate((pos_idxes_train, neg_idxes_train), axis=0)]

        valset = self.all_x[np.concatenate((pos_idxes_val, neg_idxes_val), axis=0)], \
                 self.all_y[np.concatenate((pos_idxes_val, neg_idxes_val), axis=0)]

        testset = self.all_x[np.concatenate((pos_idxes_test, neg_idxes_test), axis=0)], \
                  self.all_y[np.concatenate((pos_idxes_test, neg_idxes_test), axis=0)]

        self.mydata_cache[testfold] = (alltrainset, trainset, valset, testset)

        return alltrainset, trainset, valset, testset

    def _my_load_data(self):
        x_train, y_train = pickle.load(open('./data/MIMIC/0513-pred24/train.pkl', 'rb'))
        x_test, y_test = pickle.load(open('./data/MIMIC/0513-pred24/test.pkl', 'rb'))

        self.all_x = np.concatenate((x_train, x_test), axis=0).astype(np.float32)
        self.all_y = np.concatenate((y_train, y_test), axis=0).astype(np.int64)[:, 0]

        print('all training set shape:', self.all_x.shape)
        self.all_x = self.all_x[..., :42]
        print('Now only take first 42 features (not including missing indicators).')
        print('all training set shape:', self.all_x.shape)
        print('all label shape:', self.all_y.shape)

        self.total_pos_num = np.arange(len(self.all_x))[self.all_y == 1]
        self.total_neg_num = np.arange(len(self.all_x))[self.all_y == 0]
        np.random.shuffle(self.total_pos_num)
        np.random.shuffle(self.total_neg_num)

    def _load_pytorch_loader(self, testfold=4, feature_idxes=None, subset=True):
        alltrainset, trainset, valset, testset = self._load_data(testfold=testfold)

        def subset_features(theset):
            x, y = theset
            x = x[:, :, feature_idxes]
            return x, y

        def zero_feature(theset):
            x, y = theset
            rest_features = np.delete(np.arange(0, x.shape[1]), feature_idxes)
            copy_x = copy.deepcopy(x)
            copy_x[:, :, rest_features] = 0.
            return copy_x, y

        if feature_idxes is not None:
            if subset:
                trainset = subset_features(trainset)
                valset = subset_features(valset)
                testset = subset_features(testset)
            else:  # zero out features
                trainset = zero_feature(trainset)
                valset = zero_feature(valset)
                testset = zero_feature(testset)

        alltrain_loader = self._to_pytorch_loader(alltrainset)
        train_loader = self._to_pytorch_loader(trainset)
        val_loader = self._to_pytorch_loader(valset)
        test_loader = self._to_pytorch_loader(testset)
        return alltrain_loader, train_loader, val_loader, test_loader

    def init_hyperamaters(self):
        return {
            'dimensions': [42, 8, 2],
            'epochs': 100,
            'epoch_print': 1,
            'weight_lr': 5e-4,
            'lookahead': 5,
            'lr_patience': 2,
            'verbose': 1,
            'weight_decay': 1e-4,
            'loss_criteria': nn.CrossEntropyLoss(),
            'beta1': 0.5,
        }

    def init_bdnet_hyperparams(self):
        _, train_loader, val_loader, test_loader = self._load_pytorch_loader(testfold=0)
        return {
            'dropout_param_size': (1, 42),
            'reg_coef': 0.01,
            'ard_init': 0.,
            'lr': 0.001,
            'verbose': 1,
            'epochs': 10,
            'epoch_print': 1,
            'rw_max': 3,
            'loss_criteria': nn.CrossEntropyLoss(),
            'val_loader': val_loader,
            'lookahead': 10,
            'min_epochs': 60,
        }

    def get_top_indices(self):
        return [1, 2, 3, 5, 10, 20, 30, 42]

    def _get_random_sample_hyperparams(self):
        raise NotImplementedError
        # n_hidden = np.random.randint(60, 150)
        # n_layers = np.random.randint(0, 6)
        #
        # dimensions = [59, 1]
        # for i in range(n_layers):
        #     dimensions.insert(1, n_hidden)
        #
        # return {'dimensions': dimensions}

    def get_total_folds(self):
        return range(5)
