from __future__ import print_function
import os
import numpy as np
from collections import OrderedDict
import torch


class PhysionetDataHelper:
    def __init__(self, data_dir='./data/', testfold=0, valfold=1):
        assert 0 <= testfold < 10 and 0 <= valfold < 9, \
            'Invalid testfold or valfold: testfold %d, valfold %d' % (testfold, valfold)
        self.data_dir = data_dir

        self.removed_attributes = ['Height', 'Age', 'RecordID', 'ICUType', 'Gender']
        self._set_up_name_dicts()
        self.trainset, self.valset, self.testset, self.ratio_1_to_0 = \
            self._load_physionet_data(testfold, valfold)

    def _set_up_name_dicts(self):
        self.id_dict = OrderedDict()
        self.feature_dict = OrderedDict()
        the_path = os.path.join(self.data_dir, 'id_map.txt')

        with open(the_path) as fp:
            for line in fp:
                tmp = line.strip().split('\t')
                self.id_dict[tmp[0]] = int(tmp[1])
                self.feature_dict[int(tmp[1])] = tmp[0]

    def map_id_to_feature_name(self, id):
        return self.feature_dict[id]

    def map_feature_name_to_id(self, name):
        return self.id_dict[name]

    @staticmethod
    def split(x, lengths, y, testfold=0, valfold=0):
        valfold = (testfold + valfold + 1) % 10
        assert valfold != testfold, 'they are identical!'

        print('total mortality label:', y.sum())
        pos_indices = np.arange(x.size(0))[y.numpy() == 1]
        print('pos_indices shape', pos_indices.shape)
        neg_indices = np.delete(np.arange(x.size(0)), pos_indices)

        def _helper(idxes, the_x, the_lengths, the_ys):
            torch_idxes = torch.from_numpy(idxes)
            return the_x[torch_idxes], the_lengths[idxes], the_ys[torch_idxes]

        x_pos, lengths_pos, y_pos = _helper(pos_indices, x, lengths, y)
        x_neg, lengths_neg, y_neg = _helper(neg_indices, x, lengths, y)

        def _split(the_x, the_lengths, the_ys):
            fold_num = int(np.ceil(the_x.size(0) / 10))
            idxes = np.arange(the_x.size(0))

            test_idxes = np.arange(testfold * fold_num, (testfold + 1) * fold_num)
            val_idxes = np.arange(valfold * fold_num, (valfold + 1) * fold_num)
            train_idxes = np.delete(np.delete(idxes, test_idxes), val_idxes)

            trainset = _helper(train_idxes, the_x, the_lengths, the_ys)
            valset = _helper(val_idxes, the_x, the_lengths, the_ys)
            testset = _helper(test_idxes, the_x, the_lengths, the_ys)

            return trainset, valset, testset

        train_pos, val_pos, test_pos = _split(x_pos, lengths_pos, y_pos)
        train_neg, val_neg, test_neg = _split(x_neg, lengths_neg, y_neg)

        def combine_set(set1, set2):
            x1, length1, y1 = set1
            x2, length2, y2 = set2

            combined_x = torch.cat((x1, x2), dim=0)
            combined_lengths = np.concatenate((length1, length2), axis=0)
            combined_y = torch.cat((y1, y2), dim=0)
            return combined_x, combined_lengths, combined_y

        total_train = combine_set(train_pos, train_neg)
        total_lengths = combine_set(val_pos, val_neg)
        total_test = combine_set(test_pos, test_neg)

        return total_train, total_lengths, total_test

    def _load_physionet_data(self, testfold, valfold):
        trainset_path = os.path.join(self.data_dir, 'mean_missing_dataset.pth')
        x, lengths, y = torch.load(trainset_path)

        # Get relative class balance
        ratio_1_to_0 = np.sum(y == 1) * 1.0 / np.sum(y == 0.)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        trainset, valset, testset = self.split(x, lengths, y, testfold, valfold)
        return trainset, valset, testset, ratio_1_to_0

    def load_physionet_data(self, feature_idxes=None):
        '''
        Load data for sklearn classifier. So only load those important features.
        '''
        if feature_idxes is None:
            return self.trainset, self.valset, self.testset

        trainset = self.subset_features(self.trainset, feature_idxes)
        valset = self.subset_features(self.valset, feature_idxes)
        testset = self.subset_features(self.testset, feature_idxes)
        return trainset, valset, testset

    def load_test_loader(self, batch_size=16, feature_idxes=None, set_to_0=True):
        test_loader = self._load_physionet_loader(self.testset, batch_size, feature_idxes,
                                                  set_to_0)
        return test_loader

    def load_train_and_val_loader(self, batch_size=16, feature_idxes=None, set_to_0=True):
        train_loader = self._load_physionet_loader(self.trainset, batch_size,
                                                   feature_idxes, set_to_0)
        val_loader = self._load_physionet_loader(self.valset, batch_size, feature_idxes,
                                                 set_to_0)
        return train_loader, val_loader

    def load_combined_loader(self, batch_size, feature_idxes=None, set_to_0=True):
        x_train, lengths_train, y_train = self.trainset
        x_val, lengths_val, y_val = self.valset

        combined_x = torch.cat((x_train, x_val), dim=0)
        combined_lengths = np.concatenate((lengths_train, lengths_val), axis=0)
        combined_y = torch.cat((y_train, y_val), dim=0)

        return self._load_physionet_loader((combined_x, combined_lengths, combined_y),
                                           batch_size, feature_idxes, set_to_0)

    @staticmethod
    def subset_features(set, feature_idxes):
        x, lengths, y = set
        return x[:, :, torch.LongTensor(feature_idxes)], lengths, y

    @staticmethod
    def set_features_to_0(set, feature_idxes):
        x, lengths, y = set
        x_copy = x.clone()

        all_feature_idxes = np.arange(x_copy.size(2))
        rest_feature_idxes = np.delete(all_feature_idxes, feature_idxes)
        if len(rest_feature_idxes) > 0:
            x_copy[:, :, torch.LongTensor(rest_feature_idxes)] = 0.
        return x_copy, lengths, y

    @staticmethod
    def recalculate_seq_lengths(set):
        x, lengths, y = set

        lengths = np.ones(x.size(0), dtype=int)
        x_numpy = x.numpy()
        for i in range(x.shape[0]):
            for j in range(1, 49):
                if not np.all(x_numpy[i, -j, :] == 0):
                    break
            seq_length = x.shape[1] - j + 1
            lengths[i] = seq_length
        return x, lengths, y

    @staticmethod
    def _load_physionet_loader(set, batch_size=16, feature_idxes=None, set_to_0=True):
        if feature_idxes is not None:
            if not set_to_0:
                set = PhysionetDataHelper.subset_features(set, feature_idxes)
            else:
                set = PhysionetDataHelper.set_features_to_0(set, feature_idxes)
        set = PhysionetDataHelper.recalculate_seq_lengths(set)

        class Iterable(object):
            def __iter__(self):
                total_batches = int(np.ceil(set[0].size(0) * 1.0 / batch_size))
                idxes = np.arange(set[0].size(0))
                np.random.shuffle(idxes)

                for batch_idx in range(total_batches):
                    tmp = idxes[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                    yield (set[0][torch.from_numpy(tmp)], set[1][tmp],
                           set[2][torch.from_numpy(tmp)], total_batches)
        return Iterable()


class EarlyStoppingScheduler:
    def __init__(self, lookahead=5):
        self.lookahead = lookahead
        self.count_down = lookahead
        self.epoch = 0
        self.min_loss = np.inf
        self.min_epoch = None

    def is_early_stop(self, loss):
        if loss <= self.min_loss:
            self.min_loss = loss
            self.min_epoch = self.epoch
            self.count_down = self.lookahead
        else:
            self.count_down -= 1

        if self.count_down < 0:
            return True

        self.epoch += 1
        return False

if __name__ == '__main__':
    helper = PhysionetDataHelper()
    loader, _ = helper.load_physionet_loader()
    print(iter(loader).next())