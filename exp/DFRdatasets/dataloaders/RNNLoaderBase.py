import numpy as np

from .MLPLoaderBase import MLPLoaderBase


class RNNLoaderBase(MLPLoaderBase):
    '''
    Handle the 2 dim input (time, features) and to 1 dim input in sklearn.
    Especially rank function (needs to pass by num_features) and test function (take 1 dim feature).
    '''
    def get_time_feature_shape(self):
        raise NotImplementedError

    def rf_rank(self, testfold=4):
        all_metrics = super(RNNLoaderBase, self).rf_rank(testfold)
        all_metrics['rank'] = all_metrics['rank'].reshape(self.get_time_feature_shape()).mean(axis=0)
        return all_metrics

    def random_rank(self, testfold=4):
        D = self.get_time_feature_shape()[-1]
        return {'rank': np.random.rand(D)}

    def marginal_rank(self, testfold=4):
        all_metrics = super(RNNLoaderBase, self).marginal_rank(testfold)
        all_metrics['rank'] = all_metrics['rank'].reshape(self.get_time_feature_shape()).mean(axis=0)
        return all_metrics

    def enet_rank(self, testfold=4):
        all_metrics = super(RNNLoaderBase, self).enet_rank(testfold)
        all_metrics['rank'] = all_metrics['rank'].reshape(self.get_time_feature_shape()).mean(axis=0)
        return all_metrics

    def lasso_rank(self, testfold=4):
        all_metrics = super(RNNLoaderBase, self).lasso_rank(testfold)
        all_metrics['rank'] = all_metrics['rank'].reshape(self.get_time_feature_shape()).mean(axis=0)
        return all_metrics

    def nn_rank(self, testfold=4):
        all_metrics = super(RNNLoaderBase, self).nn_rank(testfold)
        all_metrics['rank'] = all_metrics['rank'][0]
        return all_metrics

    def dfs_rank(self, testfold=4):
        all_metrics = super(RNNLoaderBase, self).dfs_rank(testfold)
        all_metrics['rank'] = all_metrics['rank'][0]
        return all_metrics
