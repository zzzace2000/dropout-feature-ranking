from __future__ import print_function

import os
import sys
from os import path

sys.path.append(path.dirname(path.dirname(os.getcwd())))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def gen_simulate_data(N, P=100, n_informative=40, train_ratio=0.8, noise_var=1., test_size=200,
                      correlat_var=0., batch_size=64):

    train_loader, val_loader, gnd_truth_rank = \
        _gen_simulate_data(N, P, n_informative, train_ratio, noise_var, correlat_var, batch_size)

    test_loader, _, _ = \
        _gen_simulate_data(test_size * 2, P, n_informative, 0.5,
                           noise_var, correlat_var, batch_size)

    # Normalize them here
    all_tensor = torch.cat([l.dataset.data_tensor for l in [train_loader, val_loader, test_loader]], 0)
    total_mean = torch.mean(all_tensor, 0)
    total_std = torch.std(all_tensor, 0)

    def normalize(loader):
        dset = loader.dataset.data_tensor
        loader.dataset.data_tensor = (dset - total_mean.expand_as(dset)) / total_std.expand_as(dset)

    normalize(train_loader)
    normalize(val_loader)
    normalize(test_loader)

    return train_loader, val_loader, test_loader, gnd_truth_rank


def _gen_simulate_data(N, P=100, n_informative=40, train_ratio=0.8, noise_var=1.,
                       correlat_var=0., batch_size=64):
    # Produce datasets
    x = np.zeros((N, P)).astype('float32')
    y = np.random.randint(2, size=N)

    for i in range(N):
        mean_mat = np.array([0.01 * (f + 1) * 2 * (y[i] - 0.5) for f in range(n_informative)])
        corr_mat = (1. - correlat_var) * np.eye(n_informative) + correlat_var
        x[i, :n_informative] = np.random.multivariate_normal(mean_mat, corr_mat)

        for feature_idx in range(n_informative, P):
            x[i, feature_idx] = np.random.normal(0, noise_var)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y)

    train_num = int(train_ratio * N)

    x_train = x[:train_num, :]
    x_val = x[train_num:, :]
    y_train = y[:train_num]
    y_val = y[train_num:]

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size, shuffle=True)

    # Gnd Truth Rank
    gnd_truth_rank = np.array(list(range(P)))
    gnd_truth_rank[n_informative:] = -1

    return train_loader, val_loader, gnd_truth_rank

def vis_corrcoef(x):
    ''' (numpy 2d matrix) -> Plot a figure
    Visualize correlation coefficient of data matrix x.
    :param x: Row is the observation. Column is the data feature.
    :return: Show a correlation matrix
    '''
    corr = np.corrcoef(x, rowvar=False)
    plt.imshow(corr, interpolation='nearest')
    plt.colorbar()
    plt.show()


def cal_spearman_rank_coef(rank1, rank2):
    return spearmanr(rank1, rank2).correlation


def perturb(model, loader, perturb_val_func, num_samples=1):
    ''' (net, loader, func) -> numpy 1d array
    Perturb the input value to rank which feature is more important.
    :param model: Classifier model. Needs to have 'evaluate_loader' method
    :param loader: Data loader.
    :param perturb_val_func: Generate the perturbed tensor by 2 params:
        feature_idx: the feature index.
        size: number of samples to generate in a torch floatTensor 1d array
    :return: the perturb rank. The higher, the more important.
    '''

    assert hasattr(model, 'evaluate_loader'), 'Need to have this func!'

    N, P = loader.dataset.data_tensor.size()
    orig_loss, _ = model.evaluate_loader(loader)

    perturb_rank = np.zeros(P)
    for i in range(P):
        copy_loader = copy.deepcopy(loader)
        if num_samples > 1:
            copy_loader.dataset.data_tensor = \
                copy_loader.dataset.data_tensor.repeat(num_samples, 1)

        old_val = copy_loader.dataset.data_tensor[:, i]
        copy_loader.dataset.data_tensor[:, i] = perturb_val_func(i, old_val)

        loss, acc = model.evaluate_loader(copy_loader)

        perturb_rank[i] = loss.data[0] - orig_loss.data[0]

    return perturb_rank


def compose_loader(x, y, batch_size):
    if not hasattr(x, 'cpu'):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        y = y.view(-1, 1)

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size, shuffle=True)

    train_loader.dataset.target_tensor = train_loader.dataset.target_tensor.float()
    return train_loader

def decompose_loader(loader):
    x = loader.dataset.data_tensor.numpy()
    y = loader.dataset.target_tensor.numpy()
    return x, y

class Container:
    def __init__(self):
        '''
        :param val_name: The label of the value it returns. Ex: 'acc', 'coef'
        :param x_name: The name of x. Ex: 'time', 'datasize'
        :param line_name: The name of each legends. Ex: 'classifier'
        '''
        self.x_arr = []
        self.units = []
        self.labels = []
        self.vals = []

        self.val_name = 'a'
        self.x_name = 'b'
        self.line_name = 'c'

    def set_names(self, val_name, x_name, line_name):
        self.val_name = val_name
        self.x_name = x_name
        self.line_name = line_name

    def record_vals(self, the_dict, x_val, unit):
        keys = list(the_dict.keys())
        vals = list(the_dict.values())

        N = len(vals)

        self.x_arr += [x_val for i in range(N)]
        self.units += [unit for i in range(N)]
        self.labels += keys
        self.vals += vals

    def add_(self, container):
        self.x_arr += container.x_arr

        unit_offset = -1 if len(self.units) == 0 else max(self.units)
        self.units += [u + unit_offset + 1 for u in container.units]
        self.labels += container.labels
        self.vals += container.vals
        return self

    def add(self, container):
        newcontainer = Container()
        newcontainer.add_(self)
        newcontainer.add_(container)
        return newcontainer

    def get_pandas_table(self, allow_method_names=None):
        dataframe = pd.DataFrame([np.array(self.vals, dtype=np.float64), self.x_arr, self.units,
                                  self.labels]).transpose()
        dataframe.columns = [self.val_name, self.x_name, 'unit', self.line_name]

        if allow_method_names is not None:
            init_conditions = (dataframe[self.line_name] == allow_method_names[0])
            if len(allow_method_names) > 1:
                for name in allow_method_names[1:]:
                    init_conditions = init_conditions | (dataframe[self.line_name] == name)
            dataframe = dataframe[init_conditions]

        return dataframe

    def plot_figs(self, allow_method_names=None):
        plot_table = self.get_pandas_table(allow_method_names)

        ax = sns.tsplot(time=self.x_name, value=self.val_name,
                        unit="unit", condition=self.line_name,
                        data=plot_table, err_style="ci_bars", interpolate=True)
        return ax

    @staticmethod
    def test():
        dataframe = pd.DataFrame([[1, 2, 0], [2, 3, 0], [3, 4, 1], [4, 5, 1]])
        dataframe.columns = ['col1', 'col2', 'unit']

        ax = sns.tsplot(time='col1', value='col2',
                        unit="unit", ci="sd",
                        data=dataframe, err_style="ci_bars", interpolate=True)
        plt.show()


def run_std_err_params(param_name, values, repeat, val_func, default_params, num_output_table=1,
                       num_parallel_threads=1, kept_raw=False):
    '''
    It is used to run multiple times to get stderr and plot a figure.
    :param param_name: the x-axis name. Ex: 'time'
    :param values: array of possible x vals you want to get.
    :param repeat: How many times you repeat.
    :param val_func: Function to get the vals. Needs to accept arg 'unit'
    :param default_params: Other params need to pass in function.
    :return:
    '''
    # Run once to get all the possible parameters, then I could allocate a table
    containers = [Container() for i in range(num_output_table)]

    if num_parallel_threads == 1:
        raw = {}
        for unit in range(repeat):
            for x_val in values:
                default_params[param_name] = x_val
                default_params['unit'] = unit
                print(unit, x_val)
                outputs = val_func(**default_params)
                print(default_params)

                for i in range(num_output_table):
                    containers[i].record_vals(outputs[i], x_val, unit)

                if kept_raw:
                    raw['{}_{}'.format(x_val, unit)] = outputs
        if kept_raw:
            containers.append(raw)
    else:
        import pymp
        containers = pymp.shared.list(containers)

        with pymp.Parallel(num_parallel_threads) as p:
            for overall_idx in p.range(repeat * len(values)):
                unit = overall_idx % repeat
                val_idx = overall_idx / repeat

                default_params[param_name] = values[val_idx]
                default_params['unit'] = unit

                print(default_params)
                outputs = val_func(**default_params)

                for i in range(num_output_table):
                    with p.lock:
                        containers[i].record_vals(outputs[i], values[val_idx], unit)

    return containers


if __name__ == '__main__':
    Container.test()
    # def test_para_func(unit, test_param):
    #     return {'asd': test_param}, {'asd': test_param}
    #
    # container = run_std_err_params('test_param', ['qwe1', 'qwe2'], repeat=2, val_func=test_para_func,
    #                    default_params={}, num_parallel_threads=2, num_output_table=2)
