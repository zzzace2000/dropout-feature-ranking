import argparse
import os

import numpy as np
import torch

from dataloaders.LoaderBase import LoaderBase
import feature_utils


def run_with_test_fold(testfold, rank_name, args, loader):
    # Ex: 'nn_rank:0.01'. Then extract nn_rank and 0.01 seperately
    the_rank_func_name = rank_name
    if ':' in rank_name:
        tmp = rank_name.split(':')
        the_rank_func_name = tmp[0]

        if '|' not in tmp[1]:
            loader.bdnet_hyperparams['reg_coef'] = float(tmp[1])
        else:
            tmp2 = tmp[1].split('|')
            loader.bdnet_hyperparams['reg_coef'] = float(tmp2[0])
            dropout_rate = float(tmp2[1])
            loader.bdnet_hyperparams['ard_init'] = \
                np.log(dropout_rate) - np.log(1. - dropout_rate)

    # train a classifier and rank it. Only not nn_specific_rank
    # rank = None
    # if the_rank_func_name != 'nn_specific_rank':
    if args.no_rank_cache:
        rank = getattr(loader, the_rank_func_name)(testfold=testfold)['rank']
    else:
        folder = 'results/{}'.format(args.dataset)
        if not os.path.exists(folder):
            os.mkdir(folder)
        allfilenames = os.listdir(folder)
        the_rank_file = None
        for filename in allfilenames:
            if ('-%s-' % rank_name) in filename:
                the_rank_file = filename
                break
        if the_rank_file is None:
            rank = getattr(loader, the_rank_func_name)(testfold=testfold)['rank']
        else:
            print('rank cache found for {}, {}'.format(rank_name, testfold))
            _, cache_ranks = torch.load(os.path.join(folder, the_rank_file))
            rank = cache_ranks[testfold]

    if 'no_test' in args.test_func:
        return None, rank

    def helper_subset(unit, top_gene_num):
        indices = np.flip(rank.argsort(), axis=0).copy()
        selected_ind = indices[:top_gene_num]

        result = {}
        for the_test_func in args.test_func:
            # Not sure dict would work
            metrics = getattr(loader, the_test_func)(testfold, selected_ind)

            for attr_name in metrics:
                result['%s_%s_%s' %
                       (the_rank_func_name, the_test_func, attr_name)] \
                    = metrics[attr_name]
        return result,

    top_ind_tried = loader.get_top_indices()
    containers = feature_utils.run_std_err_params(
        'top_gene_num', top_ind_tried, repeat=1, val_func=helper_subset,
        default_params={}, num_parallel_threads=1)
    return containers, rank


def run(rank_name, args, loader):
    # Only supports 5 fold cross validation
    containers_dict, rank_dict = {}, {}
    for testfold in loader.get_total_folds():
        containers_dict[testfold], rank_dict[testfold] = run_with_test_fold(
            testfold, rank_name, args, loader)

    # Save containers and rank
    folder = 'results/{}'.format(args.dataset)
    if not os.path.exists(folder):
        os.mkdir(folder)

    filename = args.identifier + '-%s-%s' % (rank_name, '-'.join(args.test_func))
    torch.save((containers_dict, rank_dict), '{}/{}.pth'.format(folder, filename))


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
    parser.add_argument('--identifier', type=str, default='0111')
    # parser.add_argument('--reg_coef', type=float, default=None, help='vbd regularization coef!')
    parser.add_argument('--dataset', type=str, default='MIMIC_new',
                        help='["wineqaulity", "OnlineNewsPopularity", '
                             '"ClassificationONPLoader", "RegSupport2Loader"]')
    parser.add_argument('--seed', type=int, default='1234')
    # parser.add_argument('--lookahead', type=int, default=5)
    # parser.add_argument('--weighted', action='store_true', default=False)
    # parser.add_argument('--reuse-rnn', action='store_true', default=False)
    parser.add_argument('--rank_func', nargs='+', type=str,
                        default=['nn_rank:0.1'], help='specify lambda | dropout ini')
    parser.add_argument('--test_func', nargs='+', type=str, default=['nn_test_zero'],
                        help='["nn_test_zero", "nn_test_retrain"]')
    parser.add_argument('--visdom_enabled', action='store_true', default=False)
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

    # Custom change
    args.total_folds = 5
    if args.dataset == 'MIMIC':
        args.total_folds = 1

    # Custom

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
    # Get the loader
    loader = LoaderBase.create(
        args.dataset, {'visdom_enabled': args.visdom_enabled,
                       'cuda_enabled': args.cuda,
                       'nn_cache': args.nn_cache
                       })
    for rank_name in args.rank_func:
        run(rank_name, args, loader)
