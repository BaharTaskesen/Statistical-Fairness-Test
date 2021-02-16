########################################################################
# A Statistical Test for Probabilistic Fairness
# ACM FACCT 2021
# Authors: Bahar Taskesen, Jose Blanchet, Daniel Kuhn, Viet-Anh Nguyen
########################################################################
import numpy.linalg as LA
import numpy as np
import multiprocessing as mp
import math
from functools import partial
import time
gr = (math.sqrt(5) + 1) / 2
sigmoid_func = lambda beta, x: 1 / (1 + np.exp(- beta.T @ x))


def limiting_dist_EQOPP(X, a, y, beta, marginals):
    N = X.shape[0]

    coef_0 = sum([1 / N * sigmoid_func(x=x_hat, beta=beta) * (a[i]) * y[i] for i, x_hat in enumerate(X)])
    coef_1 = sum([1 / N * sigmoid_func(x=x_hat, beta=beta) * (1 - a[i]) * y[i] for i, x_hat in enumerate(X)])

    coef_ = 0
    sig = 0

    for i, x_data in enumerate(X):
        sigmoid_x = sigmoid_func(beta=beta, x=x_data)
        mean_ = 1 / N * (sigmoid_x * (marginals[0, 1] * a[i] * y[i] - marginals[1, 1] * (1 - a[i]) * y[i]) +
                        (1 - a[i]) * y[i] * coef_0 - a[i] * y[i] * coef_1)

    for i, x_data in enumerate(X):
        sigmoid_x = sigmoid_func(beta=beta, x=x_data)
        sig += 1 / N * (sigmoid_x * (marginals[0, 1] * a[i] * y[i] - marginals[1, 1] * (1 - a[i]) * y[i]) +
                        (1 - a[i]) * y[i] * coef_0 - a[i] * y[i] * coef_1 - mean_) ** 2

        coef_ += 1 / N * LA.norm(beta * (1 - sigmoid_x) * sigmoid_x * np.int(-1) ** np.int(a[i] - 1) *
                                 y[i] / marginals[np.int(a[i]), 1], ord=2) ** 2

    c = 1 / coef_ * sig / (marginals[1, 1] ** 2) / (marginals[0, 1] ** 2)

    return c


def k_opt(range_of_k, gamma, beta, data_tuple_in):
    x_hat = data_tuple_in[0]
    lambd = data_tuple_in[1]
    f_k = []
    for k in range_of_k:
        f_k.append(gamma ** 2 * lambd ** 2 * LA.norm(beta, ord=2) ** 2 * k ** 2 + \
                gamma * lambd / (1 + np.exp(gamma * lambd * LA.norm(beta, ord=2) ** 2 * k - beta.T @ x_hat)))
    min_f = np.min(f_k)
    k_opt = range_of_k[np.where(f_k == min_f)[0][0]]
    return k_opt


def function_with_k_opt(k_opts, gamma, beta, data_tuple_in):
    f_k = 0
    for i, data_tuple_i in enumerate(data_tuple_in):
        x_hat = data_tuple_i[0]
        lambd = data_tuple_i[1]
        f_k += gamma ** 2 * lambd ** 2 * LA.norm(beta, ord=2) ** 2 * k_opts[i] ** 2 + \
                       gamma * lambd / (1 + np.exp(gamma * lambd * LA.norm(beta, ord=2) ** 2 *
                                                   k_opts[i] - beta.T @ x_hat))
    return f_k



def f_gamma(gamma, k_opt, range_of_k, data, data_sensitive, data_label, beta, emp_marginals_array, parallel=True):
    '''
    :param gamma:
    :param k_opt: the function that computes optimal k
    :param range_of_k: the range that the optimal k will be computed
    :param data: empirical observation -- test data
    :param data_sensitive: empirical observation -- test sensitive attributes
    :param data_label:  empirical observation -- test data true label
    :param beta: the parameter of the given classifier
    :param emp_marginals_array: marginals of the subgroups of the empirical observation
    :return: - of the function that we use to calculate gamma --- the purpose is not to modify gsm algorithm
    '''

    # use all available CPUs
    data_arranged, data_sensitive_arranged = [], []
    for i, x_hat in enumerate(data):
        if data_label[i] == 1:
            data_arranged.append(x_hat)
            data_sensitive_arranged.append(data_sensitive[i])
    k_opt_part = partial(k_opt, range_of_k, gamma, beta)
    data_tuple_in = [[x_hat, np.int(-1) ** np.int(data_sensitive_arranged[k] - 1) / emp_marginals_array[
        int(data_sensitive_arranged[k]), 1]] for k, x_hat in enumerate(data_arranged)]
    if parallel:
        pool = mp.Pool(mp.cpu_count())
        # k_opt = [pool.apply(function_k, args=(range_of_k, gamma, 1, beta, x_hat)) for x_hat in data[data_label==1, :]]
        # k_opt = [pool.map(f_k, [x_hat.reshape([-1, 1]) for x_hat in data[data_label==1, :]])]

        k_optimum = pool.map(k_opt_part, data_tuple_in)
        pool.close()
        pool.join()
    else:
        k_optimum = []
        for t, _ in enumerate(data_tuple_in):
            k_optimum.append(k_opt_part(data_tuple_in[t]))
    res = - 1 / data.shape[0] * function_with_k_opt(k_opts=k_optimum, gamma=gamma, beta=beta, data_tuple_in=data_tuple_in)
    return res

def f_gamma_with_optimums(gamma, k_opt, range_of_k, data, data_sensitive, data_label, beta, emp_marginals_array, parallel=True):
    '''
    :param gamma:
    :param k_opt: the function that computes optimal k
    :param range_of_k: the range that the optimal k will be computed
    :param data: empirical observation -- test data
    :param data_sensitive: empirical observation -- test sensitive attributes
    :param data_label:  empirical observation -- test data true label
    :param beta: the parameter of the given classifier
    :param emp_marginals_array: marginals of the subgroups of the empirical observation
    :return: - of the function that we use to calculate gamma --- the purpose is not to modify gsm algorithm
    '''

    # use all available CPUs
    data_arranged, data_sensitive_arranged = [], []
    for i, x_hat in enumerate(data):
        if data_label[i] == 1:
            data_arranged.append(x_hat)
            data_sensitive_arranged.append(data_sensitive[i])
    k_opt_part = partial(k_opt, range_of_k, gamma, beta)
    data_tuple_in = [[x_hat, np.int(-1) ** np.int(data_sensitive_arranged[k] - 1) / emp_marginals_array[
        int(data_sensitive_arranged[k]), 1]] for k, x_hat in enumerate(data_arranged)]
    if parallel:
        pool = mp.Pool(mp.cpu_count())
        # k_opt = [pool.apply(function_k, args=(range_of_k, gamma, 1, beta, x_hat)) for x_hat in data[data_label==1, :]]
        # k_opt = [pool.map(f_k, [x_hat.reshape([-1, 1]) for x_hat in data[data_label==1, :]])]

        k_optimum = pool.map(k_opt_part, data_tuple_in)
        pool.close()
        pool.join()
    else:
        k_optimum = []
        for t, _ in enumerate(data_tuple_in):
            k_optimum.append(k_opt_part(data_tuple_in[t]))
    res = - 1 / data.shape[0] * function_with_k_opt(k_opts=k_optimum, gamma=gamma, beta=beta, data_tuple_in=data_tuple_in)
    return res, k_optimum

def get_marginals(sensitives, target):
    """Calculate marginal probabilities of test data"""
    N_test = sensitives.shape[0]
    P_11 = np.sum(
        [1 / N_test if sensitives[i] == 1 and target[i] == 1 else 0 for i
         in range(N_test)])
    P_01 = np.sum(
        [1 / N_test if sensitives[i] == 0 and target[i] == 1 else 0 for i
         in range(N_test)])
    P_10 = np.sum(
        [1 / N_test if sensitives[i] == 1 and target[i] == 0 else 0 for i
         in range(N_test)])
    P_00 = np.sum(
        [1 / N_test if sensitives[i] == 0 and target[i] == 0 else 0 for i
         in range(N_test)])
    if np.abs(P_01 + P_10 + P_11 + P_00 - 1) > 1e-10:
        print(np.abs(P_01 + P_10 + P_11 + P_00 - 1))
        print('Marginals are WRONG!')
    return P_00, P_01, P_10, P_11


def stratified_sampling(X, a, y, emp_marginals, n_train_samples):
    emp_P_11 = emp_marginals[1, 1]
    emp_P_01 = emp_marginals[0, 1]
    emp_P_10 = emp_marginals[1, 0]
    emp_P_00 = emp_marginals[0, 0]
    X_11, X_01, X_10, X_00 = [], [], [], []
    for i in range(X.shape[0]):
        if a[i] == 1 and y[i] == 1:
            X_11.append(X[i, :])
        if a[i] == 0 and y[i] == 1:
            X_01.append(X[i, :])
        if a[i] == 1 and y[i] == 0:
            X_10.append(X[i, :])
        if a[i] == 0 and y[i] == 0:
            X_00.append(X[i, :])
    ind_11 = np.random.randint(low=0, high=np.array(X_11).shape[0], size=int(emp_P_11 * n_train_samples))
    ind_01 = np.random.randint(low=0, high=np.array(X_01).shape[0], size=int(emp_P_01 * n_train_samples))
    ind_10 = np.random.randint(low=0, high=np.array(X_10).shape[0], size=int(emp_P_10 * n_train_samples))
    ind_00 = np.random.randint(low=0, high=np.array(X_00).shape[0], size=int(emp_P_00 * n_train_samples))
    X_train_11 = np.array(X_11)[ind_11, :]
    X_train_01 = np.array(X_01)[ind_01, :]
    X_train_10 = np.array(X_10)[ind_10, :]
    X_train_00 = np.array(X_00)[ind_00, :]
    X_test11 = np.delete(np.array(X_11), ind_11, axis=0)
    X_test01 = np.delete(np.array(X_01), ind_01, axis=0)
    X_test10 = np.delete(np.array(X_10), ind_10, axis=0)
    X_test00 = np.delete(np.array(X_00), ind_00, axis=0)
    test_sensitives = np.hstack([[1] * X_test11.shape[0], [0] * X_test01.shape[0],
                                 [1] * X_test10.shape[0], [0] * X_test00.shape[0]])
    y_test = np.hstack([[1] * X_test11.shape[0], [1] * X_test01.shape[0],
                        [0] * X_test10.shape[0], [0] * X_test00.shape[0]])
    X_test = np.vstack([X_test11, X_test01, X_test10, X_test00])
    y_train = np.hstack([[1] * int(emp_P_11 * n_train_samples), [1] * int(emp_P_01 * n_train_samples),
                         [0] * int(emp_P_10 * n_train_samples), [0] * int(emp_P_00 * n_train_samples)])
    train_sensitives = np.hstack([[1] * int(emp_P_11 * n_train_samples), [0] * int(emp_P_01 * n_train_samples),
                                  [1] * int(emp_P_10 * n_train_samples), [0] * int(emp_P_00 * n_train_samples)])
    X_train = np.vstack([X_train_11, X_train_01, X_train_10, X_train_00])

    threshold = 1 - sum(y == 1) / y.shape[0]
    return X_train, train_sensitives, y_train, X_test, test_sensitives, y_test, threshold


def gss(f, a, b, tol):
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c

        # We recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2


def calculate_distance_prob_eqopp(data_tuple, function_of_gamma, range_gamma, k_opt_fcn, range_of_k, beta, tol):
    data = data_tuple[0]
    data_sensitive = data_tuple[1]
    data_label = data_tuple[2]

    emp_marginals = np.reshape(get_marginals(sensitives=data_sensitive, target=data_label), [2, 2])

    f_gamma_part = partial(function_of_gamma,
                           k_opt=k_opt_fcn,
                           range_of_k=range_of_k, data=data,
                           data_sensitive=data_sensitive,
                           data_label=data_label, beta=beta,
                           emp_marginals_array=emp_marginals)

    gamma_opt = gss(f=f_gamma_part, a=range_gamma[0], b=range_gamma[1], tol=tol)

    distance = - f_gamma_part(gamma_opt)

    return distance


def calculate_distance_prob_eqopp_with_opt_params(data_tuple, function_of_gamma, range_gamma, k_opt_fcn, range_of_k, beta, tol):
    data = data_tuple[0]
    data_sensitive = data_tuple[1]
    data_label = data_tuple[2]

    emp_marginals = np.reshape(get_marginals(sensitives=data_sensitive, target=data_label), [2, 2])

    f_gamma_part = partial(function_of_gamma,
                           k_opt=k_opt_fcn,
                           range_of_k=range_of_k, data=data,
                           data_sensitive=data_sensitive,
                           data_label=data_label, beta=beta,
                           emp_marginals_array=emp_marginals)

    gamma_opt = gss(f=f_gamma_part, a=range_gamma[0], b=range_gamma[1], tol=tol)
    distance, optimum_ks = f_gamma_with_optimums(gamma=gamma_opt, k_opt=k_opt_fcn, range_of_k=range_of_k, data=data,
                          data_sensitive=data_sensitive, data_label=data_label, beta=beta,
                          emp_marginals_array=emp_marginals)
    distance = - distance
    # distance = - f_gamma_part(gamma_opt)

    return distance, optimum_ks, gamma_opt