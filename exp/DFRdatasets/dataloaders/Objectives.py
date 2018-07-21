from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR, LinearSVR, LinearSVC
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import ttest_ind, pearsonr
import glmnet_python
from glmnet import glmnet; from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict


class Regression:
    def lasso_rank(self, trainset, testset):
        return self._lasso_enet_common(
            trainset, testset, alpha=1., name='lasso')

    def enet_rank(self, trainset, testset):
        return self._lasso_enet_common(
            trainset, testset, alpha=0.5, name='enet')

    def _lasso_enet_common(self, trainset, testset, alpha, name):
        x_train, y_train = trainset
        x_test, y_test = testset
        new_x_train = x_train.copy().astype(np.float64)
        new_y_train = y_train.copy().astype(np.float64)
        fit = glmnet(x=new_x_train.copy(), y=new_y_train.copy(), family='gaussian',
                     alpha=alpha, nlambda=1000)

        def _get_rank_by_soln_path(soln_path_coefs):
            rank = np.zeros(soln_path_coefs.shape[0])

            for f in range(soln_path_coefs.shape[0]):
                for i in range(soln_path_coefs.shape[1]):
                    if soln_path_coefs[f, i] != 0.:
                        rank[f] = -i
                        break

            rank[rank == 0] = -(soln_path_coefs.shape[1])
            return rank

        rank = _get_rank_by_soln_path(fit['beta'])

        # call glmnet get acc
        cvfit = cvglmnet(x=new_x_train.copy(), y=new_y_train.copy(),
                         alpha=alpha, family='gaussian')
        test_pred = cvglmnetPredict(cvfit, newx=x_test, s='lambda_min')
        test_abs_error = ((y_test - test_pred) ** 2).mean()
        print('{} test error (L2): {}'.format(name, test_abs_error))
        return rank, {'loss': test_abs_error}

    def marginal_rank(self, trainset, testset):
        '''
        Basically just pearson corelation for each feature
        '''
        x_train, y_train = trainset
        if y_train.ndim == 2:
            y_train = y_train.ravel()

        r_squares = np.ones(x_train.shape[1])
        for i in range(x_train.shape[1]):
            pearson_corr, _ = pearsonr(x_train[:, i], y_train)
            r_squares[i] = pearson_corr ** 2

        # The smaller pvalue, it means it has higher rank
        return r_squares

    def mim_rank(self, trainset, testset):
        raise NotImplementedError('Impossible to do in regresion settings.')

    def rf_rank(self, trainset, testset):
        clf = RandomForestRegressor(n_estimators=200, n_jobs=4)
        test_abs_error, clf = self._sklearn_test(clf, 'rf', trainset, testset)
        return clf.feature_importances_, {'loss': test_abs_error}

    def svm_rbf_test(self, trainset, testset, feature_idxes=None):
        clf = SVR()
        test_abs_error, clf = self._sklearn_test(clf, 'svm-rbf', trainset, testset, feature_idxes)
        return {'loss': test_abs_error}

    def svm_linear_test(self, trainset, testset, feature_idxes=None):
        clf = LinearSVR()
        test_abs_error, clf = self._sklearn_test(clf, 'svm-linear', trainset, testset, feature_idxes)
        return {'loss': test_abs_error}

    def _sklearn_test(self, clf, clf_name, trainset, testset, feature_idxes=None):
        x_train, y_train = trainset
        x_test, y_test = testset

        if feature_idxes is not None:
            x_train = x_train[:, feature_idxes]
            x_test = x_test[:, feature_idxes]

        if y_train.ndim == 2:
            y_train = y_train.ravel()
            y_test = y_test.ravel()
        # Create a random forest classifier. By convention, clf means 'classifier'
        clf.fit(x_train, y_train)

        pred = clf.predict(x_test)
        test_abs_error = ((y_test - pred) ** 2).mean()
        print('{} test error (L2): {}'.format(clf_name, test_abs_error))

        return test_abs_error, clf


class Classification:
    def lasso_rank(self, trainset, testset):
        return self._lasso_enet_common(
            trainset, testset, alpha=1., name='lasso')

    def enet_rank(self, trainset, testset):
        return self._lasso_enet_common(
            trainset, testset, alpha=0.5, name='enet')

    def _lasso_enet_common(self, trainset, testset, alpha, name):
        x_train, y_train = trainset
        x_test, y_test = testset
        new_x_train = x_train.copy().astype(np.float64)
        new_y_train = y_train.copy().astype(np.float64)
        fit = glmnet(x=new_x_train.copy(), y=new_y_train.copy(), family='binomial',
                     alpha=alpha, nlambda=1000)

        def _get_rank_by_soln_path(soln_path_coefs):
            rank = np.zeros(soln_path_coefs.shape[0])

            for f in range(soln_path_coefs.shape[0]):
                for i in range(soln_path_coefs.shape[1]):
                    if soln_path_coefs[f, i] != 0.:
                        rank[f] = -i
                        break

            rank[rank == 0] = -(soln_path_coefs.shape[1])
            return rank

        rank = _get_rank_by_soln_path(fit['beta'])

        # call glmnet get acc
        cvfit = cvglmnet(x=new_x_train.copy(), y=new_y_train.copy(),
                         alpha=alpha, family='binomial', ptype='class')
        test_pred = cvglmnetPredict(cvfit, newx=x_test, s='lambda_min', ptype='class')
        acc = (test_pred[:, 0] == y_test).sum() * 1.0 / y_test.shape[0]

        test_prob = cvglmnetPredict(cvfit, newx=x_test, s='lambda_min', ptype='response')

        test_auroc = sklearn.metrics.roc_auc_score(y_test, test_prob, average='macro')
        test_aupr = sklearn.metrics.average_precision_score(y_test, test_prob,
                                                            average='macro')
        print(name, 'testacc:', acc, 'test_auroc:', test_auroc, 'test_aupr:', test_aupr)
        return rank, {'auroc': test_auroc, 'aupr': test_aupr, 'acc': acc}

    def marginal_rank(self, trainset, testset):
        '''
        Basically just ttest for each feature
        '''
        x_train, y_train = trainset
        assert (y_train > 1).sum() == 0, 'Only 2 class is support.' + str(y_train)

        x_train_0 = x_train[y_train == 0]
        x_train_1 = x_train[y_train == 1]

        pvalues = np.ones(x_train.shape[1])
        for i in range(x_train.shape[1]):
            _, pvalue = ttest_ind(x_train_0[:, i], x_train_1[:, i])
            pvalues[i] = pvalue

        # The smaller pvalue, it means it has higher rank
        return -pvalues

    def mim_rank(self, trainset, testset):
        ''' Mutial information Maximization (MIM) '''
        x_train, y_train = trainset
        assert (y_train > 1).sum() == 0, 'Only 2 class is support.' + str(y_train)

        from sklearn.feature_selection import mutual_info_classif
        return mutual_info_classif(x_train, y_train)

    def rf_rank(self, trainset, testset):
        clf = RandomForestClassifier(n_estimators=200, n_jobs=4)
        clf, metrics = self._sklearn_test(clf, 'rf', trainset, testset)
        return clf.feature_importances_, metrics

    def svm_rbf_test(self, trainset, testset, feature_idxes=None):
        clf = SVC(probability=True)
        _, metrics = self._sklearn_test(
            clf, 'svm-rbf', trainset, testset, feature_idxes)
        return metrics

    def svm_linear_test(self, trainset, testset, feature_idxes=None):
        clf = SVC(kernel='linear', probability=True)
        _, metrics = self._sklearn_test(
            clf, 'svm-linear', trainset, testset, feature_idxes)
        return metrics

    def _sklearn_test(self, clf, clf_name, trainset, testset, feature_idxes=None):
        x_train, y_train = trainset
        x_test, y_test = testset

        if feature_idxes is not None:
            x_train = x_train[:, feature_idxes]
            x_test = x_test[:, feature_idxes]
        if y_train.ndim == 2:
            y_train = y_train.ravel()
            y_test = y_test.ravel()

        # Create a random forest classifier. By convention, clf means 'classifier'
        clf.fit(x_train, y_train)

        pred_test = clf.predict(x_test)
        testacc = np.sum(pred_test == y_test) * 1.0 / pred_test.shape[0]

        pred_test = clf.predict_proba(x_test)
        test_auroc = sklearn.metrics.roc_auc_score(y_test, pred_test[:, 1], average='macro')
        test_aupr = sklearn.metrics.average_precision_score(y_test, pred_test[:, 1],
                                                            average='macro')
        print(clf_name, 'testacc:', testacc, 'test_auroc:', test_auroc, 'test_aupr:', test_aupr)
        return clf, {'auroc': test_auroc, 'aupr': test_aupr, 'acc': testacc}

    # def svc_rbf_test(self, args, loadhelper, feature_idxes=None):
    #     clf = SVC(probability=True)
    #     return _sklearn_test(args, loadhelper, clf, 'svc-rbf', feature_idxes)
    #
    # def svc_linear_test(args, loadhelper, feature_idxes=None):
    #     clf = SVC(kernel='linear', probability=True)
    #     return _sklearn_test(args, loadhelper, clf, 'svc-linear', feature_idxes)
    #
    # def rf_test(args, loadhelper, feature_idxes=None):
    #     clf = RandomForestClassifier(n_estimators=5000)
    #     return _sklearn_test(args, loadhelper, clf, 'rf', feature_idxes)
    #
    # def _sklearn_test(args, loadhelper, clf, clf_name, feature_idxes=None):
    #     trainset, valset, testset = loadhelper.load_physionet_data(feature_idxes=feature_idxes)
    #
    #     def append_missing_features(set):
    #         x, _, y = set
    #         x = x.numpy()
    #         y = y.numpy().ravel()
    #         missing = np.zeros(x.shape)
    #         missing[x == 0.] = 1.
    #         newx = np.concatenate((x, missing), axis=-1)
    #         newx = newx.reshape((newx.shape[0], -1))
    #         return newx, y
    #
    #     x_train, y_train = append_missing_features(trainset)
    #     x_val, y_val = append_missing_features(valset)
    #     x_combined = np.concatenate((x_train, x_val), axis=0)
    #     y_combined = np.concatenate((y_train, y_val), axis=0)
    #
    #     clf.fit(x_combined, y_combined)
    #
    #     x_test, y_test = append_missing_features(testset)
    #     pred_test = clf.predict(x_test)
    #     testacc = np.sum(pred_test == y_test) * 1.0 / pred_test.shape[0]
    #
    #     pred_test = clf.predict_proba(x_test)
    #     test_auroc = sklearn.metrics.roc_auc_score(y_test, pred_test[:, 1], average='macro')
    #     test_aupr = sklearn.metrics.average_precision_score(y_test, pred_test[:, 1],
    #                                                         average='macro')
    #     print(clf_name, 'testacc:', testacc, 'test_auroc:', test_auroc, 'test_aupr:', test_aupr)
    #     return clf, test_auroc, test_aupr, clf_name
