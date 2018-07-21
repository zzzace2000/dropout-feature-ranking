import random
import numpy as np

from mimic3models import nn_utils
from mimic3models import metrics
from mimic3models.in_hospital_mortality import utils
from mimic3benchmark.readers import InHospitalMortalityReader
from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics

import numpy as np
import pandas as pd
import torch.nn as nn
import copy
from .RNNLoaderBase import RNNLoaderBase
from exp.DFRdatasets.models.LSTM import ClfLSTM


class MIMIC(RNNLoaderBase):
    '''
    Data specific loader to define nn_test and nn_rank.
    ::: test func: return {name, val}
    ::: rank func: return {rank}
    '''
    def __init__(self, **kwargs):
        kwargs['mode'] = 'classification'
        super(MIMIC, self).__init__(**kwargs)
        self.total_folds = 1
        self.nn_class = ClfLSTM

    def get_time_feature_shape(self):
        return 12, 76

    def load_data(self, testfold=0):
        ''' For sklearn module. So only return train and test set.'''
        assert testfold == 0, 'No cross validation here!'
        train_raw, val_raw, test_raw = super(MIMIC, self).load_data(testfold=0)

        # reshape as a numpy array.
        def reshape_x(set):
            x, y = set
            x = x.reshape((x.shape[0], -1))
            return x, y
        train_raw = reshape_x(train_raw)
        test_raw = reshape_x(test_raw)
        return train_raw, None, None, test_raw

    def _load_data(self, testfold=4):
        train_reader = InHospitalMortalityReader(
            dataset_dir='mimic3-benchmarks/data/in-hospital-mortality/train/',
            listfile='mimic3-benchmarks/data/in-hospital-mortality/train_listfile.csv',
            period_length=48.0)

        val_reader = InHospitalMortalityReader(
            dataset_dir='mimic3-benchmarks/data/in-hospital-mortality/train/',
            listfile='mimic3-benchmarks/data/in-hospital-mortality/val_listfile.csv',
            period_length=48.0)

        test_reader = InHospitalMortalityReader(
            dataset_dir='mimic3-benchmarks/data/in-hospital-mortality/test/',
            listfile='mimic3-benchmarks/data/in-hospital-mortality/test_listfile.csv',
            period_length=48.0)

        discretizer = Discretizer(timestep=float(4),
                                  store_masks=True,
                                  imput_strategy='previous',
                                  start_time='zero')

        discretizer_header = discretizer.transform(
            train_reader.read_example(0)[0])[1].split(',')
        cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

        normalizer = Normalizer(fields=cont_channels)  # choose here onlycont vs all
        normalizer.load_params('mimic3-benchmarks/mimic3models/in_hospital_mortality/'
                               'ihm_ts%s.input_str:%s.start_time:zero.normalizer'
                               % ('2.0', 'previous'))
        # normalizer=None

        train_raw = utils.load_data(train_reader, discretizer, normalizer, False)
        val_raw = utils.load_data(val_reader, discretizer, normalizer, False)
        test_raw = utils.load_data(test_reader, discretizer, normalizer, False)

        # To split into
        def preprocess(the_raw_set):
            x, y = the_raw_set
            x = x.astype(np.float32, copy=False)
            y = np.array(y)
            return x, y

        train_raw = preprocess(train_raw)
        val_raw = preprocess(val_raw)
        test_raw = preprocess(test_raw)
        return train_raw, val_raw, test_raw

    def _load_pytorch_loader(self, testfold=4, feature_idxes=None, subset=True):
        assert testfold == 0

        trainset, valset, testset = super(MIMIC, self).load_data(testfold=0)

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

        train_loader = self._to_pytorch_loader(trainset, batch_size=64)
        val_loader = self._to_pytorch_loader(valset, batch_size=64)
        test_loader = self._to_pytorch_loader(testset, batch_size=64)
        return train_loader, val_loader, test_loader

    def init_hyperamaters(self):
        return {
            'dimensions': [76, 256, 2],
            'epochs': 20,
            'epoch_print': 1,
            'weight_lr': 1e-4,
            'lookahead': 3,
            'lr_patience': 1,
            'verbose': 1,
            'weight_decay': 0.,
            'loss_criteria': nn.CrossEntropyLoss(),
            'beta1': 0.5,
        }

    def init_bdnet_hyperparams(self):
        train_loader, val_loader, test_loader = self._load_pytorch_loader(testfold=0)
        return {
            'dropout_param_size': (1, 76),
            'reg_coef': 0.01,
            'ard_init': 0.,
            'lr': 0.002,
            'verbose': 1,
            'epochs': 200,
            'epoch_print': 1,
            'rw_max': 30,
            'loss_criteria': nn.CrossEntropyLoss(),
            'val_loader': val_loader,
            'lookahead': 10,
            'min_epochs': 60,
        }

    def get_top_indices(self):
        return [1, 2, 3, 5, 10, 20, 35, 50, 76]

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
        return range(1)
