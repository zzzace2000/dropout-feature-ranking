import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.MLP import MLP
import torch.nn as nn
from dataloaders.LinearSimulation import LinearSimulation
import scipy
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt


def plot_scatter(x, y, xlabel, ylabel):
    pearson = pearsonr(x, y)
    plt.scatter(x, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title('Pearson: %.3f' % pearson[0])

    plt.show()


if __name__ == '__main__':
    loader = LinearSimulation(
        mode='regression', n_informative=50, N=200, P=100,
        noise_var=0.5, weight_magnitude=1.,
        visdom_enabled=False, cuda_enabled=False, nn_cache=False)

    # result = loader.nn_rank()
    result = loader.nn_joint_rank(testfold=4)

    gnd_truth_weight = loader.get_gnd_truth_weight()
    the_learned_weight = result['model'].trained_classifier.ffnn[0].weight.data.cpu().numpy()[0, :]

    x = gnd_truth_weight[:, 0]

    plot_scatter(x, the_learned_weight, 'Gnd Truth Weight', 'Learned Weight')

    keep_prob = torch.sigmoid(torch.FloatTensor(result['rank'])).numpy()
    plot_scatter(x, keep_prob, 'Gnd Truth Weight', 'Keep Prob')

    product = the_learned_weight * keep_prob
    plot_scatter(x, product, 'Gnd Truth Weight', 'Product of dropout and weights')

    lasso_result = loader.lasso_rank()
    plot_scatter(x, lasso_result['rank'], 'Gnd Truth Weight', 'LASSO rank')

    enet_result = loader.enet_rank()
    plot_scatter(x, enet_result['rank'], 'Gnd Truth Weight', 'Elastic Net rank')

    # Calculate loss. With dropout rate or not
    alltrainset, trainset, valset, testset = loader.load_data(testfold=4)

    x_test, y_test = testset

    test_pred = np.dot(x_test, np.expand_dims(the_learned_weight, axis=-1))
    orig_loss = ((y_test - test_pred) ** 2).mean()

    test_pred = np.dot(
        x_test, np.expand_dims(the_learned_weight * keep_prob, axis=-1))
    with_dropout_loss = ((y_test - test_pred) ** 2).mean()

    the_learned_weight[keep_prob < 0.01] = 0
    my_pred = np.dot(x_test, np.expand_dims(the_learned_weight, axis=-1))
    my_trick_loss = ((y_test - my_pred) ** 2).mean()

    print('VBD: orig loss vs dropout loss:', orig_loss, with_dropout_loss,
          my_trick_loss)

    gnd_truth_loss = np.mean(np.square(y_test - np.dot(
        x_test, gnd_truth_weight)))
    print('Gnd_truth_loss:', gnd_truth_loss)

    print('Done')
    # {'rank': nn_rank, 'metrics': test_metrics, 'model': vbdnet}

