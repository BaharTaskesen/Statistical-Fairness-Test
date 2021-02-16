"""
Logistic Regression
"""
########################################################################
# A Statistical Test for Probabilistic Fairness
# ACM FACCT 2021
# Authors: Bahar Taskesen, Jose Blanchet, Daniel Kuhn, Viet-Anh Nguyen
########################################################################
import numpy as np
import cvxpy as cp
from collections import namedtuple
from sklearn.metrics import log_loss
from functions import get_marginals


class LogisticRegression():
    def __init__(self, fit_intercept=True, reg=0, verbose=False):
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.reg = reg

    def fit(self, X, y):
        """
                Fit the model according to the given training data.
                Parameters
                ----------
                X : {array-like, sparse matrix} of shape (n_samples, n_features)
                    Training vector, where n_samples is the number of samples and
                    n_features is the number of features.
                y : array-like of shape (n_samples,)
                    Target vector relative to X.
        """
        dim = X.shape[1]
        beta = cp.Variable(dim)  # coeefficients
        b = cp.Variable(1)  # intercept
        t = cp.Variable(1)
        if self.fit_intercept:
            log_likelihood = cp.sum(
                cp.multiply(y, X @ beta + b) - cp.logistic(X @ beta + b))
        else:
            log_likelihood = cp.sum(
                cp.multiply(y, X @ beta) - cp.logistic(X @ beta))

        cons = []
        if self.reg:
            cons.append(cp.SOC(t, beta))
        else:
            cons.append(t == 0)

        self.problem = cp.Problem(cp.Maximize(log_likelihood / dim - self.reg * t), cons)
        self.problem.solve(solver=cp.ECOS, abstol=1e-15, verbose=False)
        self.coef_ = beta.value
        if self.fit_intercept:
            self.intercept_ = b.value
        else:
            self.intercept_ = 0


    def predict_proba(self, X):
        """
            Probability estimates.
            The returned estimates for the class are ordered by the
            label of classes.
            For a multi_class problem, if multi_class is set to be "multinomial"
            the softmax function is used to find the predicted probability of
            each class.
            Else use a one-vs-rest approach, i.e calculate the probability
            of each class assuming it to be positive using the logistic function.
            and normalize these values across all the classes.
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Vector to be scored, where `n_samples` is the number of samples and
                `n_features` is the number of features.
            Returns
            -------
            T : array-like of shape (n_samples, 2)
                Returns the probability of the sample for each class in the model,
                where classes are ordered as they are in ``self.classes_``.
        """
        proba = np.zeros((X.shape[0], 2))
        h = 1 / (1 + np.exp(- self.coef_ @ X.T - self.intercept_))
        proba[:, 1] = h
        proba[:, 0] = 1 - h
        return proba

    def predict_log_proba(self, X):
        """
                Predict logarithm of probability estimates.
                The returned estimates for all classes are ordered by the
                label of classes.
                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    Vector to be scored, where `n_samples` is the number of samples and
                    `n_features` is the number of features.
                Returns
                -------
                T : array-like of shape (n_samples, n_classes)
                    Returns the log-probability of the sample for each class in the
                    model, where classes are ordered as they are in ``self.classes_``.
        """
        prob_clip = np.clip(self.predict_proba(X), a_min=1e-15, a_max=1 - 1e-15)
        return np.log(prob_clip)

    def predict(self, X, thr=0.5):
        """
            Predict class of covariates.
            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                Vector to be scored, where `n_samples` is the number of samples and
                `n_features` is the number of features.
            thr : scale
                threshold to predict the class of given data x such that if predict_proba(x)>= thr the predicted
                class is 1 and otherwise the predicted class is 0.
            Returns
            -------
            T : vector of shape (n_classes, )
                Returns the predicted classes of the samples
        """
        probs = self.predict_proba(X)[:, 1]
        probs[probs >= thr] = 1
        probs[probs< thr] = 0
        probs_bin = probs
        return probs_bin

    def logloss(self, X, y):
        return log_loss(y_true=y, y_pred=self.predict_proba(X)[:, 1])

    def score(self, X, y, thr=0.5):
        # calculate accuracy of the given test data set
        predictions = self.predict(X, thr)
        N = X.shape[0]
        TP = np.sum([1 if predictions[i] == 1 and y[i] == 1 else 0 for i in range(N)])
        TN = np.sum([1 if predictions[i] == 0 and y[i] == 0 else 0 for i in range(N)])
        FP = np.sum([1 if predictions[i] == 1 and y[i] == 0 else 0 for i in range(N)])
        FN = np.sum([1 if predictions[i] == 0 and y[i] == 1 else 0 for i in range(N)])
        mean_acc = (TP + TN) / (TP + TN + FP + FN)
        return mean_acc

    def precision(self, X, y, thr=0.5):
        # calculate accuracy of the given test data set
        predictions = self.predict(X, thr)
        N = X.shape[0]
        TP = np.sum([1 if predictions[i] == 1 and y[i] == 1 else 0 for i in range(N)])
        TN = np.sum([1 if predictions[i] == 0 and y[i] == 0 else 0 for i in range(N)])
        FP = np.sum([1 if predictions[i] == 1 and y[i] == 0 else 0 for i in range(N)])
        FN = np.sum([1 if predictions[i] == 0 and y[i] == 1 else 0 for i in range(N)])
        precision = TP / (TP + FP)
        return precision

    def unfairness(self, X, a, y, thr=0.5):
        N = X.shape[0]
        _, P_01, _, P_11 = get_marginals(a, y)
        E_lh11 = []
        E_lh01 = []
        E_h11 = []
        E_h01 = []
        P_111 = 0
        P_001 = 0
        probs = self.predict_proba(X)[:, 1]
        log_probs = self.predict_log_proba(X)[:, 1]
        predictions = self.predict(X, thr)
        for i in range(N):
            if a[i] == 1 and y[i] == 1:
                E_lh11.append(log_probs[i])
                E_h11.append(probs[i])
                if predictions[i] == 1:
                    P_111 += 1 / P_11 / N
            if a[i] == 0 and y[i] == 1:
                E_lh01.append(log_probs[i])
                E_h01.append(probs[i])
                if predictions[i] == 0:
                    P_001 += 1 / P_01 / N

        E_h11_mean = np.mean(E_h11)
        E_h01_mean = np.mean(E_h01)
        E_lh01_mean = np.mean(E_lh01)
        E_lh11_mean = np.mean(E_lh11)

        prob_unfairness = np.abs(E_h11_mean - E_h01_mean)  # Probabilistic Unfairness
        log_prob_unfairness = np.abs(E_lh11_mean - E_lh01_mean)  # Log-Probabilistic Unfairness
        det_unfairness = np.abs(P_111 + P_001 - 1)  # Deterministic Unfairness

        UnfairnessMeasures = namedtuple('UnfairnessMeasures', 'det_unfairness, prob_unfairness, log_prob_unfairness')(det_unfairness, prob_unfairness, log_prob_unfairness)
        return UnfairnessMeasures
