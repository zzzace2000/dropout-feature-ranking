import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from arch.sensitivity.BDNet import L1BDNet, SigmoidBDNet, MiddleBDNet, TrainWeightsL1BDNet
from exp.DFRdatasets.models.MLP import MLP, ClassificationMLP
from .LoaderBase import LoaderBase
import os, sys
import copy
from arch.sensitivity.DFSNet import DFSNet

def logit(x):
    return np.log(x) - np.log(1. - x)


class MLPLoaderBase(LoaderBase):
    def __init__(self, mode, nn_cache=True, **kwargs):
        super(MLPLoaderBase, self).__init__(mode=mode, **kwargs)
        self.hyper_params = self.init_hyperamaters()
        self.bdnet_hyperparams = self.init_bdnet_hyperparams()
        self.nn_cache = nn_cache

        if mode == 'regression':
            self.nn_class = MLP
        else:
            self.nn_class = ClassificationMLP

        self.vbdnet = L1BDNet
        self.batch_size = 128
        self.vbd_net_callback = None # For specific metrics

    def init_hyperamaters(self):
        raise NotImplementedError

    def init_bdnet_hyperparams(self):
        raise NotImplementedError

    def _load_data(self, testfold=4):
        raise NotImplementedError

    def get_top_indices(self):
        raise NotImplementedError

    def _get_random_sample_hyperparams(self):
        raise NotImplementedError

    def random_sample_hyperparams(self):
        sample_hyperparams = self._get_random_sample_hyperparams()
        self.hyper_params.update(sample_hyperparams)

    def nn_rank(self, testfold=4):
        assert 0 <= testfold < 5, 'testfold out of range. testfold: ' + str(testfold)

        # Train neural network
        nn, test_metrics = self.get_nn(testfold)

        # Rank nn feature based on var dropout
        vbdnet = self._nn_rank(nn, testfold, L1BDNet)
        nn_rank = -vbdnet.logit_p.data.cpu().numpy()
        return {'rank': nn_rank, 'metrics': test_metrics, 'model': vbdnet}

    def nn_joint_rank(self, testfold=4):
        ''' Jointly train dropout rates and weights '''
        nn = self.nn_class(dimensions=self.hyper_params['dimensions'],
                           loss_criteria=self.hyper_params['loss_criteria'])

        assert 'weights_lr' in self.bdnet_hyperparams
        vbdnet = self._nn_rank(nn, testfold, TrainWeightsL1BDNet)
        nn_rank = -vbdnet.logit_p.data.cpu().numpy()

        all_train_loader, train_loader, val_loader, test_loader = self._load_pytorch_loader(testfold)

        test_acc, test_loss = vbdnet.eval_loader_with_loss_and_acc(test_loader, cuda=self.cuda_enabled)
        test_metrics = {'loss': test_loss, 'acc': test_acc,
                        # 'auroc': auroc, 'aupr': aupr
                        }
        print('nn joint rank test:', str(test_metrics))
        return {'rank': nn_rank, 'metrics': test_metrics, 'model': vbdnet}

    def nn_specific_rank(self, testfold=4, top_gene_num=None):
        # Train neural network
        nn, test_metrics = self.get_nn(testfold)

        self.bdnet_hyperparams['num_features'] = top_gene_num
        # self.bdnet_hyperparams['ard_init'] = \
        #     logit(top_gene_num * 1.0 / self.hyper_params['dimensions'][0])
        vbdnet = self._nn_rank(nn, testfold, SigmoidBDNet)
        nn_rank = -vbdnet.logit_p.data.cpu().numpy()
        return {'rank': nn_rank}

    def nn_middle_rank(self, testfold=4):
        # Train neural network
        nn, test_metrics = self.get_nn(testfold)
        vbdnet = nn_rank = self._nn_rank(nn, testfold, MiddleBDNet)
        nn_rank = -vbdnet.logit_p.data.cpu().numpy()
        return {'rank': nn_rank}

    def get_nn(self, testfold=4):
        '''
        Handle caching of nn
        '''
        if not self.nn_cache:
            new_nn, test_metrics = self._nn_train(testfold)
        else:
            if not os.path.exists('nn_cache'):
                os.mkdir('nn_cache')
            cache_dir = os.path.join('nn_cache', self.__class__.__name__)
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            file_path = os.path.join(cache_dir, '%d.pth' % testfold)
            if not os.path.exists(file_path):
                new_nn, test_metrics = self._nn_train(testfold)
                torch.save(new_nn, file_path)
                # when training a neural network, record the metrics
                metric_dict = {'testfold': testfold, 'cls': 'nn'}
                metric_dict.update(test_metrics)
                self.record_metrics_to_tsv(metric_dict)
            else:
                nn = torch.load(file_path, map_location=lambda storage, loc: storage)
                new_nn = self.nn_class(
                    dimensions=self.hyper_params['dimensions'],
                    loss_criteria=self.hyper_params['loss_criteria'])
                new_nn.load_state_dict(nn.state_dict())
                if self.cuda_enabled:
                    new_nn.cuda()

                _, _, _, test_loader = self._load_pytorch_loader(testfold)
                test_metrics = new_nn.test_eval_loader(test_loader,
                                                       cuda=self.cuda_enabled)
        return new_nn, test_metrics

    def _nn_train(self, testfold=4, feature_idxes=None):
        _, train_loader, val_loader, test_loader = self._load_pytorch_loader(testfold, feature_idxes)

        if feature_idxes is not None:
            self.hyper_params['dimensions'][0] = len(feature_idxes)

        nn = self.nn_class(dimensions=self.hyper_params['dimensions'],
                           loss_criteria=self.hyper_params['loss_criteria'])
        nn.fit(train_loader, val_loader, cuda=self.cuda_enabled, **self.hyper_params)
        test_metrics = nn.test_eval_loader(test_loader, cuda=self.cuda_enabled)
        print('mlp test:', str(test_metrics))

        return nn, test_metrics

    def _nn_rank(self, nn, testfold=4, bdnet_class=L1BDNet):
        all_train_loader, train_loader, val_loader, test_loader = self._load_pytorch_loader(testfold)

        if 'dropout_param_size' not in self.bdnet_hyperparams:
            P = train_loader.dataset.data_tensor.size()[-1]
            self.bdnet_hyperparams['dropout_param_size'] = (P,)

        vbd_net = bdnet_class(trained_classifier=nn,
                              cuda_enabled=self.cuda_enabled,
                              **self.bdnet_hyperparams)

        if self.visdom_enabled:
            rank = torch.sigmoid(-vbd_net.logit_p.data).numpy()
            title = 'NN {:.2e} ({})'.format(self.bdnet_hyperparams['reg_coef'], testfold)
            thewin = self._vis_plot_rank(
                rank, title=title)

            def visdom_func():
                # assert self.vis.win_exists(win)
                rank = torch.sigmoid(-vbd_net.logit_p.cpu().data).numpy()
                self._vis_plot_rank(rank, title=title,
                                    thewin=thewin)
                return ''

            vbd_net.register_epoch_callback(visdom_func)

        if self.vbd_net_callback is not None:
            self.vbd_net_callback(vbd_net)

        vbd_net.fit(all_train_loader, **self.bdnet_hyperparams)

        return vbd_net

    def zero_rank(self, testfold=4):
        '''
        Use 'zero' heuristics to rank features. (basically just leave-one-out)
        :param testfold: the testfold. Between 0 to 4.
        :param rank_idxes: use these indexes to rank things.
        :return:
        '''
        # Train neural network
        nn, _ = self.get_nn(testfold)

        def zero(feature_ind, oldval):
            return torch.zeros(oldval.size())

        # Rank nn feature based on var dropout
        nn_rank = self._perturb_rank(nn, zero, testfold)
        if self.visdom_enabled:
            self._vis_plot_rank(nn_rank, 'Zero (%d)' % testfold)
        return {'rank': nn_rank}

    def shuffle_rank(self, testfold=4):
        # Train neural network
        nn, _ = self.get_nn(testfold)

        def shuffle(feature_ind, oldval):
            return oldval[torch.randperm(oldval.size(0))]

        # Rank nn feature based on var dropout
        nn_rank = self._perturb_rank(nn, shuffle, testfold)
        if self.visdom_enabled:
            self._vis_plot_rank(nn_rank, 'Shuffle (%d)' % testfold)
        return {'rank': nn_rank}

    def iter_zero_rank(self, testfold=4):
        def zero(feature_ind, oldval):
            return torch.zeros(oldval.size())
        return self._iter_perturb_rank(testfold, zero)

    def iter_shuffle_rank(self, testfold=4):
        def shuffle(feature_ind, oldval):
            return oldval[torch.randperm(oldval.size(0))]
        return self._iter_perturb_rank(testfold, shuffle)

    def _iter_perturb_rank(self, testfold, perturb_func):
        # Rank nn feature based on var dropout
        nn, _ = self.get_nn(testfold)
        rank_idxes = np.arange(0, self.hyper_params['dimensions'][0])
        results = np.zeros(self.hyper_params['dimensions'][0])

        for i in range(self.hyper_params['dimensions'][0]):
            nn_rank = self._perturb_rank(nn, perturb_func, testfold, rank_idxes)
            the_top_most_rank = rank_idxes[np.argmax(nn_rank)]
            results[the_top_most_rank] = -i

            rank_idxes = np.delete(rank_idxes, np.argmax(nn_rank))

        return {'rank': results}

    def _perturb_rank(self, nn, perturb_val_func, testfold, rank_idxes=None):
        ''' (net, loader, func) -> numpy 1d array
        Perturb the input value to rank which feature is more important.
        :param nn: Classifier model. Needs to have 'test_eval_loader' method
        :param perturb_val_func: Generate the perturbed tensor by 2 params:
            feature_idx: the feature index.
            size: number of samples to generate in a torch floatTensor 1d array
        :param rank_idxes: only the indexes need to be ranked.
        :return: the perturb rank. The higher, the more important.
        '''

        _, train_loader, val_loader, test_loader = self._load_pytorch_loader(testfold)

        P = train_loader.dataset.data_tensor.size()[-1]
        orig_metrics = nn.test_eval_loader(train_loader, self.cuda_enabled)

        if rank_idxes is None:
            rank_idxes = np.arange(0, P)

        perturb_rank = np.zeros(len(rank_idxes))
        for i, rank_idx in enumerate(rank_idxes):
            copy_loader = copy.deepcopy(train_loader)

            old_val = copy_loader.dataset.data_tensor[..., rank_idx]
            copy_loader.dataset.data_tensor[..., rank_idx] = perturb_val_func(rank_idx, old_val)

            perturb_metrics = nn.test_eval_loader(copy_loader, cuda=self.cuda_enabled)

            perturb_rank[i] = perturb_metrics['loss'] - orig_metrics['loss']

        return perturb_rank

    def dfs_rank(self, testfold=4):
        '''
        https://link.springer.com/chapter/10.1007/978-3-319-16706-0_20

        Use L1 penalty and a single 1-to-1 layer and return the magnitude of the weights as rank.
        :return: rank
        '''
        # Train neural network
        nn, _ = self.get_nn(testfold)

        _, train_loader, val_loader, test_loader = self._load_pytorch_loader(testfold)
        if 'dropout_param_size' not in self.bdnet_hyperparams:
            P = train_loader.dataset.data_tensor.size()[-1]
            self.bdnet_hyperparams['dropout_param_size'] = (P,)

        dfsnet = DFSNet(self.bdnet_hyperparams['dropout_param_size'], nn,
                        lr=self.bdnet_hyperparams['lr'],
                        l1_reg_coef=self.bdnet_hyperparams['reg_coef'],
                        loss_criteria=self.bdnet_hyperparams['loss_criteria'],
                        cuda_enabled=self.cuda_enabled, verbose=True)
        dfsnet.fit(train_loader, **self.bdnet_hyperparams)

        return {'rank': dfsnet.map_weights.data.cpu().numpy()}

    def nn_test_retrain(self, testfold=4, feature_idxes=None):
        nn, test_metrics = self._nn_train(testfold, feature_idxes)
        return test_metrics

    def nn_test_zero(self, testfold=4, feature_idxes=None):
        nn, test_metrics = self.get_nn(testfold)

        # zero out the rest of features!
        all_train_loader, train_loader, val_loader, test_loader = \
            self._load_pytorch_loader(testfold, feature_idxes, subset=False)

        test_metrics = nn.test_eval_loader(test_loader, cuda=self.cuda_enabled)
        return test_metrics

    def nn_test_hyperparams(self, testfold=4):
        _, test_metrics = self._nn_train(testfold)
        return test_metrics

    def _to_pytorch_loader(self, dataset):
        x, y = dataset
        torch_x = torch.from_numpy(x)
        torch_y = torch.from_numpy(y)

        loader = DataLoader(
            TensorDataset(torch_x, torch_y),
            batch_size=self.batch_size, shuffle=True)
        return loader

    def _load_pytorch_loader(self, testfold=4, feature_idxes=None, subset=True):
        alltrainset, trainset, valset, testset = self.load_data(testfold)

        def subset_features(theset):
            x, y = theset
            x = x[:, feature_idxes]
            return x, y

        def zero_feature(theset):
            x, y = theset
            rest_features = np.delete(np.arange(0, x.shape[1]), feature_idxes)
            copy_x = copy.deepcopy(x)
            copy_x[:, rest_features] = 0.
            return copy_x, y

        if feature_idxes is not None:
            if subset:
                alltrainset = subset_features(alltrainset)
                trainset = subset_features(trainset)
                valset = subset_features(valset)
                testset = subset_features(testset)
            else: # zero out features
                alltrainset = zero_feature(alltrainset)
                trainset = zero_feature(trainset)
                valset = zero_feature(valset)
                testset = zero_feature(testset)

        all_train_loader = self._to_pytorch_loader(alltrainset)
        train_loader = self._to_pytorch_loader(trainset)
        val_loader = self._to_pytorch_loader(valset)
        test_loader = self._to_pytorch_loader(testset)
        return all_train_loader, train_loader, val_loader, test_loader
