from __future__ import print_function

from sklearn.ensemble import RandomForestRegressor
import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from dataloaders.WineQaulityLoader import WineQaulityLoader
from dataloaders.OnlineNewsPopularityLoader import OnlineNewsPopularityLoader
from dataloaders.Support2Loader import Support2Loader

import numpy as np
import torch
import exp.feature.feature_utils as feature_utils
from collections import OrderedDict
from train_utils import get_data_loader_by_dataset


def write_to_tsv(hyper_params, test_mse_dict, path):
    '''
    :param test_mse_dict: {testfold: {'loss': 1.2, 'acc': 0.96}}
    '''

    def get_array_of_key(key):
        the_arr = []
        for testfold in test_mse_dict:
            the_arr.append(test_mse_dict[testfold][key])
        return the_arr

    result_dict = OrderedDict()
    keys = test_mse_dict[next(iter(test_mse_dict))].keys()
    for k in keys:
        values = get_array_of_key(k)
        mean_values = np.mean(values)
        std_values = np.std(values)

        result_dict['mean_%s' % k] = str(mean_values)
        result_dict['std_%s' % k] = str(std_values)
        result_dict['%s_arr' % k] = str(values)

    is_initiated = os.path.exists(path)
    with open(path, 'a') as op:
        if not is_initiated:
            print('\t'.join(list(result_dict.keys()) + ['hyperparams']), file=op)
        print('\t'.join(list(result_dict.values()) + [str(hyper_params)]), file=op)


def run_with_test_fold(testfold, loader):
    test_metrics = loader.nn_test_hyperparams(testfold)
    return test_metrics


def run(args):
    loader = get_data_loader_by_dataset(args.dataset, {'cuda_enabled': args.cuda})

    # Random sample a hyperparams
    loader.random_sample_hyperparams()

    # Only supports 5 fold cross validation
    test_mse_dict = {}
    for testfold in range(3):
        test_mse_dict[testfold] = run_with_test_fold(testfold, loader)

    # Save containers and rank
    folder = 'hyperparams/'
    if not os.path.exists(folder):
        os.mkdir(folder)

    hyper_params = loader.hyper_params
    write_to_tsv(hyper_params, test_mse_dict, 'hyperparams/{}.tsv'.format(args.dataset))



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='train rnn to predict')
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--reg_coef', type=float, default=0.001)
    # parser.add_argument('--batch-size', type=int, default=32)
    # parser.add_argument('--batch-print', type=int, default=30)
    # parser.add_argument('--save-freq', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[],
                        help='number of gpus to produce')
    parser.add_argument('--identifier', type=str, default='0105')
    # parser.add_argument('--mode', type=str, default='cv', help='["cv", "subset"]')
    parser.add_argument('--dataset', type=str, default='YearMSD',
                        help='["wineqaulity", "OnlineNewsPopularity"]')
    parser.add_argument('--seed', type=int, default='1234')
    # parser.add_argument('--lookahead', type=int, default=5)
    # parser.add_argument('--weighted', action='store_true', default=False)
    # parser.add_argument('--reuse-rnn', action='store_true', default=False)
    parser.add_argument('--rank_func', type=str, default='joint_rank')
    parser.add_argument('--test_func', type=str, default='svm_rbf_test')
    # parser.add_argument('--use-combined', action='store_true', default=False)
    # parser.add_argument('--start_val', type=int, default=2)

    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        print('gpu current device:', torch.cuda.current_device())
        torch.cuda.manual_seed(args.seed)
        if len(args.gpu_ids) > 0:
            print('start using gpu device:', args.gpu_ids)
            torch.cuda.set_device(args.gpu_ids[0])

    args.identifier += '-%s-%s' % (args.rank_func, args.test_func)

    print('args:', args)
    print('==================== Start =====================')
    print('')
    return args


if __name__ == '__main__':
    args = parse_args()

    for i in range(100):
        run(args)
