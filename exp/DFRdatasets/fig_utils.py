import torch
import numpy as np
from collections import OrderedDict
import seaborn as sns
import os
import matplotlib.pyplot as plt
import scipy
import matplotlib


def cal_rnn_rf_sig_in_table(the_table, table_name='rnn', num_features=1):
    return cal_sig_in_table(the_table, table_name=table_name, compared='rf', num_features=num_features)


def cal_rnn_zero_sig_in_table(the_table, table_name='rnn', num_features=1):
    return cal_sig_in_table(the_table, table_name=table_name, compared='zero', num_features=num_features)


def cal_sig_in_table(the_table, table_name='rnn', compared='rf', num_features=1):
    tmp = the_table[the_table['Methods'] == 'rnn_rank_%s_aupr' % table_name]
    tmp = tmp[tmp['Number of features'] == num_features]

    tmp2 = the_table[the_table['Methods'] == '%s_rank_%s_aupr' % (compared, table_name)]
    tmp2 = tmp2[tmp2['Number of features'] == num_features]

    if len(tmp) == 0 or len(tmp2) == 0:
        return 1., 0.

    max_value = max(max(tmp['Test AUPR']), max(tmp['Test AUPR']))

    #     return scipy.stats.wilcoxon(tmp['Test AUPR'], tmp2['Test AUPR']).pvalue, max_value
    return scipy.stats.ranksums(tmp['Test AUPR'], tmp2['Test AUPR']).pvalue, max_value

#     return scipy.stats.ttest_ind(tmp['Test AUPR'], tmp2['Test AUPR']).pvalue, max_value
#     return scipy.stats.ttest_rel(tmp['Test AUPR'], tmp2['Test AUPR']).pvalue, max_value


def change_name_table(table):
    # Change the names of Methods to display in the figure
    rank_orders = ['nn_rank', 'nn_joint_rank', 'nn_middle_rank', 'nn_specific_rank',
                   'dfs_rank', 'marginal_rank', 'mim_rank', 'zero_rank',
                   'shuffle_rank', 'rf_rank', 'enet_rank', 'lasso_rank', 'random_rank']
    rank_names = ['Dropout FR', 'Joint FR', 'Middle FR', 'Specific FR', 'Deep FS',
                  'Marginal', 'MIM', 'Mean', 'Shuffle',
                  'Random Forest', 'ElasticNet', 'LASSO', 'Random']
    test_methods = ['nn_test_zero', 'nn_test_retrain', 'svm_linear_test']
    metrics = ['loss', 'auroc', 'aupr']

    names_dict = OrderedDict()
    for rank_idx, rank in enumerate(rank_orders):
        for test_method in test_methods:
            for metric in metrics:
                name = '_'.join([rank, test_method, metric])
                value = rank_names[rank_idx]
                names_dict[name] = value

    table['order'] = -1
    for idx, key in enumerate(names_dict):
        table.loc[table[table['Methods'] == key].index, 'order'] = idx
        table.loc[table[table['Methods'] == key].index, 'Methods'] = names_dict[key]
    table.sort_values('order', inplace=True)
    return table


def plot_figs(container, table_name, ax=None, fig=None, title=None, test_wilcoxn=False):
    allow_method_names = None
    if table_name is not None:
        allow_method_names = ['%s_%s' % (k, table_name)
                              for k in ['nn_rank', 'nn_joint_rank', 'nn_middle_rank', 'nn_specific_rank',
                                        'zero_rank', 'random_rank',
                                        'rf_rank', 'shuffle_rank',
                                        'marginal_rank', 'mim_rank', 'dfs_rank', 'enet_rank',
                                        'lasso_rank', 'iter_zero_rank', 'iter_shuffle_rank']]
    plot_table = container.get_pandas_table(allow_method_names)

    if test_wilcoxn:
        for sig_method, symbol, height in [(cal_rnn_rf_sig_in_table, '*', 0.45),
                                           # (cal_rnn_zero_sig_in_table, 'o', 0.48)
                                           ]:
            for nf in [1, 2, 5, 10, 15, 20, 30, 37]:
                pvalue, max_value = sig_method(plot_table, table_name, num_features=nf)
                if pvalue < 0.1:  # 1-side test
                    print(pvalue, nf, table_name)
                    ax.text(nf, height, symbol, ha='center', va='bottom', )

    # sort the table and change the name
    plot_table = change_name_table(plot_table)

    if ax is None:
        fig, ax = plt.subplots()
    ax = sns.tsplot(time=container.x_name, value=container.val_name, ax=ax,
                    unit="unit", condition=container.line_name,
                    data=plot_table, err_style="ci_bars", interpolate=True)

    # get the best line plot
    max_feature_val = plot_table[plot_table[container.x_name] ==
                                 plot_table[container.x_name].max()]
    max_feature_val = np.mean(max_feature_val[container.val_name])

    ax.axhline(max_feature_val, linestyle='--', color='black', alpha=0.3)

    if title is not None:
        ax.set_title(title)

    return fig, ax


def sum_all_the_dict(dicts):
    lengths = [len(d) for d in dicts]
    min_idx = np.argmin(lengths)
    assert len(dicts[min_idx]) == min(lengths)

    print('smallest length of dictionary: %d with length %d' % (min_idx, len(dicts[min_idx])))

    first_key = next(iter(dicts[0]))
    init_container = dicts[0][first_key][0]
    for idx, the_dict in enumerate(dicts):
        #         for key in the_dict:
        for key in dicts[min_idx]:
            if idx == 0 and key == first_key:
                continue
            init_container = init_container.add(the_dict[key][0])
    return init_container


def sum_all_container_by_filenames(filenames, dataset):
    tmp = []
    for f in filenames:
        path = os.path.join('../results',dataset, f)
        container_dicts, _ = torch.load(path)
        tmp.append(container_dicts)
    return sum_all_the_dict(tmp)

