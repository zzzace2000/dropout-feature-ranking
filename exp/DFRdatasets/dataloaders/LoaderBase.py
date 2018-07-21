from __future__ import print_function

import os

import numpy as np
from sklearn.linear_model import LinearRegression

from .Objectives import Regression, Classification
import visdom


class LoaderBase(object):
    def __init__(self, mode='regression', visdom_enabled=False,
                 cuda_enabled=False, **kwargs):
        self.mode = mode
        self.sklearn_module = Regression() if mode == 'regression' else Classification()
        self.cache_dataset = {}
        self.cuda_enabled = cuda_enabled
        self.visdom_enabled = visdom_enabled

        if self.visdom_enabled:
            self.vis = visdom.Visdom()

    @staticmethod
    def create(dataset, attributes={}):
        # Get the loader
        if dataset == 'wineqaulity':
            from .WineQaulityLoader import WineQaulityLoader
            loader = WineQaulityLoader(**attributes)
        elif dataset == 'support2':
            from .Support2Loader import Support2Loader, RegSupport2Loader
            loader = Support2Loader(**attributes)
        elif dataset == 'OnlineNewsPopularity':
            from .OnlineNewsPopularityLoader import OnlineNewsPopularityLoader
            loader = OnlineNewsPopularityLoader(**attributes)
        elif dataset == 'ClassificationONPLoader':
            from .OnlineNewsPopularityLoader import ClassificationONPLoader
            loader = ClassificationONPLoader(**attributes)
        elif dataset == 'RegSupport2Loader':
            from .Support2Loader import Support2Loader, RegSupport2Loader
            loader = RegSupport2Loader(**attributes)
        elif dataset == 'YearMSD':
            from .YearMSD import YearMSD
            loader = YearMSD(**attributes)
        elif dataset == 'MIMIC':
            from .MIMIC import MIMIC
            loader = MIMIC(**attributes)
        elif dataset == 'MiniBooNE':
            from .MiniBooNE import MiniBooNE
            loader = MiniBooNE(**attributes)
        elif dataset == 'HIGGS':
            from .HIGGS import HIGGS
            loader = HIGGS(**attributes)
        elif dataset == 'FMA':
            from .FMA import FMA
            loader = FMA(**attributes)
        elif dataset == 'CreditCard':
            from .CreditCard import CreditCard
            loader = CreditCard(**attributes)
        elif dataset == 'InteractionSimulation':
            from .InteractionSimulation import InteractionSimulation
            loader = InteractionSimulation(**attributes)
        elif dataset == 'NoInteractionSimulation':
            from .InteractionSimulation import NoInteractionSimulation, MoreInteractionSimulation
            loader = NoInteractionSimulation(**attributes)
        elif dataset == 'MoreInteractionSimulation':
            from .InteractionSimulation import NoInteractionSimulation, MoreInteractionSimulation
            loader = MoreInteractionSimulation(**attributes)
        elif dataset == 'CorrelatedInteractionSimulation':
            from .InteractionSimulation import CorrelatedInteractionSimulation
            loader = CorrelatedInteractionSimulation(**attributes)
        elif dataset == 'CorrelatedNoInteractionSimulation':
            from .InteractionSimulation import CorrelatedNoInteractionSimulation
            loader = CorrelatedNoInteractionSimulation(**attributes)
        elif dataset == 'GaussSimulation':
            from .GaussSimulation import GaussSimulation
            loader = GaussSimulation(**attributes)
        elif dataset == 'ARCENE':
            from .ARCENE import ARCENE
            loader = ARCENE(**attributes)
        elif dataset == 'MADELON':
            from .MADELON import MADELON
            loader = MADELON(**attributes)
        elif dataset == 'GISETTE':
            from .GISETTE import GISETTE
            loader = GISETTE(**attributes)
        elif dataset == 'DEXTER':
            from .DEXTER import DEXTER
            loader = DEXTER(**attributes)
        elif dataset == 'DOROTHEA':
            from .DOROTHEA import DOROTHEA
            loader = DOROTHEA(**attributes)
        elif dataset == 'MIMIC_new':
            from.MIMIC_new import MIMIC_new
            loader = MIMIC_new(**attributes)
        else:
            raise Exception('No such dataset!' + dataset)
        return loader


    def load_data(self, testfold=4):
        if testfold not in self.cache_dataset:
            self.cache_dataset[testfold] = self._load_data(testfold)
        return self.cache_dataset[testfold]

    def _load_data(self, testfold=4):
        raise NotImplementedError

    def clear_cache(self):
        self.cache_dataset.clear()

    def get_top_indices(self):
        raise NotImplementedError

    def nn_rank(self, testfold=4):
        raise NotImplementedError

    def rf_rank(self, testfold=4):
        alltrainset, trainset, valset, testset = self.load_data(testfold)
        rank, metrics = self.sklearn_module.rf_rank(alltrainset, testset)

        metric_dict = {'testfold': testfold, 'cls': 'rf'}
        metric_dict.update(metrics)
        self.record_metrics_to_tsv(metric_dict)

        if self.visdom_enabled:
            self._vis_plot_rank(rank, 'RF (%d)' % testfold)

        ret_dict = {'rank': rank}
        ret_dict.update(metrics)
        return ret_dict

    def _vis_plot_rank(self, rank, title, thewin=None):
        if thewin is None:
            win = self.vis.line(Y=rank, X=np.arange(0, rank.shape[0]), opts=dict(
                xlabel='Features',
                ylabel='Importance',
                title=title,
            ))
            return win
        else:
            self.vis.line(Y=rank, X=np.arange(0, rank.shape[0]),
                          opts={'title': title},
                          win=thewin, update='replace')

    def zero_rank(self, testfold=4):
        raise NotImplementedError

    def random_rank(self, testfold=4):
        alltrainset, trainset, valset, testset = self.load_data(testfold)
        D = alltrainset[0].shape[1]
        return {'rank': np.random.rand(D)}

    def marginal_rank(self, testfold=4):
        alltrainset, trainset, valset, testset = self.load_data(testfold)
        rank = self.sklearn_module.marginal_rank(alltrainset, testset)
        if self.visdom_enabled:
            self._vis_plot_rank(rank, 'Marginal (%d)' % testfold)
        return {'rank': rank}

    def mim_rank(self, testfold=4):
        alltrainset, trainset, valset, testset = self.load_data(testfold)
        rank = self.sklearn_module.mim_rank(alltrainset, testset)
        if self.visdom_enabled:
            self._vis_plot_rank(rank, 'MIM (%d)' % testfold)
        return {'rank': rank}

    def enet_rank(self, testfold=4):
        alltrainset, trainset, valset, testset = self.load_data(testfold)
        rank, metrics = self.sklearn_module.enet_rank(alltrainset, testset)

        metric_dict = {'testfold': testfold, 'cls': 'enet'}
        metric_dict.update(metrics)
        self.record_metrics_to_tsv(metric_dict)

        ret_dict = {'rank': rank}
        ret_dict.update(metrics)
        return ret_dict

    def lasso_rank(self, testfold=4):
        alltrainset, trainset, valset, testset = self.load_data(testfold)
        rank, metrics = self.sklearn_module.lasso_rank(alltrainset, testset)

        metric_dict = {'testfold': testfold, 'cls': 'lasso'}
        metric_dict.update(metrics)
        self.record_metrics_to_tsv(metric_dict)

        ret_dict = {'rank': rank}
        ret_dict.update(metrics)
        return ret_dict

    def svm_rbf_test(self, testfold=4, feature_idxes=None):
        alltrainset, trainset, valset, testset = self.load_data(testfold)
        return self.sklearn_module.svm_rbf_test(alltrainset, testset, feature_idxes)

    def svm_linear_test(self, testfold=4, feature_idxes=None):
        alltrainset, trainset, valset, testset = self.load_data(testfold)
        return self.sklearn_module.svm_linear_test(alltrainset, testset, feature_idxes)

    @staticmethod
    def split_valset(dataset, ratio=0.1):
        '''
        Split the validation set by certain ratio percentage.
        :param dataset: (x, y)
        :param ratio: Ex: 0.1 means 10% split as validation set
        :return: trainset, valset
        '''
        x, y = dataset

        val_num = int(x.shape[0] * (1. - ratio))
        x_train = x[:val_num]
        x_val = x[val_num:]
        y_train = y[:val_num]
        y_val = y[val_num:]
        return (x_train, y_train), (x_val, y_val)

    @staticmethod
    def split_train_and_test(dataset, testfold=4):
        '''
        Split dataset as 5 fold .
        :param dataset: (x_tensor, y_tensor). Numpy array for boths.
        :param testfold: 0~4. Indicates which fold to return
        :return:
        '''
        assert 0 <= testfold < 5, 'testfold out of range: ' + str(testfold)
        x, y = dataset
        N = x.shape[0]

        fold_num = int(N * 1.0 / 5)
        x_train = np.delete(x, range(testfold * fold_num, (testfold + 1) * fold_num), axis=0)
        x_test = x[testfold * fold_num:(testfold + 1) * fold_num]

        y_train = np.delete(y, range(testfold * fold_num, (testfold + 1) * fold_num), axis=0)
        y_test = y[testfold * fold_num:(testfold + 1) * fold_num]

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_ols_error(trainset, testset):
        x_train, y_train = trainset
        x_test, y_test = testset
        lr_model = LinearRegression()
        lr_model.fit(x_train, y_train)

        pred = lr_model.predict(x_test)
        test_abs_error = ((y_test - pred) ** 2).mean()
        print('linear regression test abs error:', test_abs_error)
        return test_abs_error

    def record_metrics_to_tsv(self, test_mse_dict):
        '''
            :param test_mse_dict: {testfold: 4, cls: rf, aupr: 0.71, auroc: 0.91}
        '''

        path = 'results/%s.tsv' % self.__class__.__name__
        is_initiated = os.path.exists(path)
        with open(path, 'a') as op:
            # if not is_initiated:
            print('\t'.join(test_mse_dict.keys()), file=op)
            print('\t'.join([str(v) for v in test_mse_dict.values()]), file=op)

    def get_total_folds(self):
        return range(5)
