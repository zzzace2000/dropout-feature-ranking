import argparse
import argparse
import os

import numpy as np
import torch

from dataloaders.LoaderBase import LoaderBase
import exp.feature.feature_utils as feature_utils


def run_with_identifier(unit, corr_val, datasize, rank_names, loader, show_ols=True):
    loader.clear_cache()

    # Ex: 'nn_rank:0.01'. Then extract nn_rank and 0.01 seperately
    result = {}
    performance = {}
    ranks = {}
    for rank_name in rank_names:
        the_rank_func_name = rank_name
        if ':' in rank_name:
            tmp = rank_name.split(':')
            the_rank_func_name = tmp[0]
            loader.bdnet_hyperparams['reg_coef'] = float(tmp[1])

        # Run different datasizes / correlations.
        # Ex: datasizes => [100_0, 200_0, 1000]
        # Ex: correlations => [1000_0.1, 1000_0.2]
        identifier = '%d_%f' % (datasize, corr_val)

        # return a dictionary but not a rank!!!
        rank_dict = getattr(loader, the_rank_func_name)(testfold=identifier)
        if 'metrics' in rank_dict:
            for attr_name in rank_dict['metrics']:
                performance['%s_%s' % (the_rank_func_name, attr_name)] = rank_dict

        metrics = loader.evaluate(rank_dict['rank'])
        for attr_name in metrics:
            result['%s_%s' % (the_rank_func_name, attr_name)] = metrics[attr_name]

        ranks['%s_rank' % the_rank_func_name] = rank_dict['rank']

    if show_ols:
        performance['ols'] = loader.get_ols_error()

    return result, performance, ranks

def run(mode, args):
    # Get the loader
    loader = LoaderBase.create(
        args.dataset, {'visdom_enabled': args.visdom_enabled,
                       'cuda_enabled': args.cuda,
                       'nn_cache': args.nn_cache
                       })

    default_params = {
        'datasize': 1000, 'corr_val': -1,
        'rank_names': args.rank_func, 'show_ols': False,
        'loader': loader,
    }
    if mode == 'correlation':
        corr_vals = np.arange(0., 1.0, 0.1)
        # corr_vals = [0., 0.1]
        containers = feature_utils.run_std_err_params(
            'corr_val', values=corr_vals, repeat=args.repeat, val_func=run_with_identifier,
            default_params=default_params, num_output_table=2, kept_raw=True)
    else:
        datasizes = [100, 200, 1000, 3000]
        containers = feature_utils.run_std_err_params(
            'datasize', values=datasizes, repeat=args.repeat, val_func=run_with_identifier,
            default_params=default_params, num_output_table=2, kept_raw=True)

    raw = containers.pop()
    # Save containers and rank
    folder = 'results/{}'.format(args.dataset)
    if not os.path.exists(folder):
        os.mkdir(folder)

    filename = args.identifier + '-%s' % mode
    torch.save(containers, '{}/{}.pth'.format(folder, filename))
    torch.save(raw, '{}/{}_raw.pth'.format(folder, filename))


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
    parser.add_argument('--gpu-ids', nargs='+', type=int, default=[0],
                        help='number of gpus to produce')
    parser.add_argument('--identifier', type=str, default='0201')
    # parser.add_argument('--reg_coef', type=float, default=None, help='vbd regularization coef!')
    parser.add_argument('--dataset', type=str, default='GaussSimulation',
                        help='["wineqaulity", "OnlineNewsPopularity", '
                             '"ClassificationONPLoader", "RegSupport2Loader"]')
    parser.add_argument('--seed', type=int, default='1234')
    parser.add_argument('--repeat', type=int, default=1)
    # parser.add_argument('--lookahead', type=int, default=5)
    # parser.add_argument('--weighted', action='store_true', default=False)
    # parser.add_argument('--reuse-rnn', action='store_true', default=False)
    parser.add_argument('--modes', nargs='+', type=str,
                        default=['correlation'], help='correlation / sizes')
    parser.add_argument('--rank_func', nargs='+', type=str,
                        default=['vbd_linear_rank'], help='nn_rank')
    # parser.add_argument('--test_func', nargs='+', type=str, default=['no_test'],
    #                     help='["nn_test_zero", "nn_test_retrain"]')
    parser.add_argument('--visdom_enabled', action='store_true', default=True)
    parser.add_argument('--no_rank_cache', action='store_true', default=False)
    parser.add_argument('--no_nn_cache', action='store_true', default=False)
    # parser.add_argument('--start_val', type=int, default=2)

    args = parser.parse_args()
    args.nn_cache = (not args.no_nn_cache)
    args.rank_cache = (not args.no_rank_cache)
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        print('gpu current device:', torch.cuda.current_device())
        torch.cuda.manual_seed(args.seed)
        if len(args.gpu_ids) > 0:
            print('start using gpu device:', args.gpu_ids)
            torch.cuda.set_device(args.gpu_ids[0])

    print('args:', args)
    print('==================== Start =====================')
    print('')
    return args


if __name__ == '__main__':
    args = parse_args()

    if 'other_ranks' in args.rank_func:
        args.rank_func.remove('other_ranks')
        args.rank_func += ['marginal_rank', 'rf_rank', 'zero_rank',
                           'shuffle_rank', 'random_rank', 'enet_rank', 'lasso_rank']
    for mode in args.modes:
        run(mode, args)
