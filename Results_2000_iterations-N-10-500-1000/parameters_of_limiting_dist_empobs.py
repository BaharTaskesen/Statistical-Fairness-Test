from load_data import *
# from matplotlib.colors import ListedColormap
# from matplotlib import colors as mcolors
# import matplotlib.pyplot as plt  # for plotting stuff
from random import seed, shuffle
# import os
from functions import *
import time
# import matplotlib.pyplot as plt
# import scipy.special as sps
import datetime
from scipy.stats import gamma
# import matplotlib
# from matplotlib import rcParams


def sigmoid_func(beta, x):
    return 1 / (1 + np.exp(- beta.T @ x))


if __name__ == "__main__":
    # LOAD DATASET
    ds = 'toy_correlated'
    # n_data = 1e6
    tol = 1e-10
    true_marginals = np.array(([.4, .1], [.3, .2])) ## np.array(([P_00, P_01], [P_10, P_11]))
    beta = np.array([0, 1])
    range_gamma = [-5, 5]
    range_of_k = np.linspace(0, 1/8, 100)
    replications = 2000
    N_range = [10, 500, 1000] #[10, 100, 1000, 5000]
    #########
    min_rej_rates = np.zeros([replications, len(N_range)])
    print('True marginals=' + str(true_marginals))
    print('range of gamma:' + str(range_gamma))
    print('range of k:' + 'from' + str(min(range_of_k)) + ' to ' + str(max(range_of_k)) + ' with ' + str(range_of_k.shape)
          + ' number of steps')
    print('Replications=' + str(replications))
    print('Beta=' + str(beta))
    print('N_range=' + str(N_range))
    print('Tolerance for gamma:' + str(tol))
    thetas = np.zeros([replications, len(N_range)])
    for i, N in enumerate(N_range):
        SEED = 1122334455
        seed(SEED)
        np.random.seed(SEED)
        rng = np.random.RandomState(2)
        for rep in range(replications):
            start = time.time()

            marginals_rand = np.random.multinomial(N, true_marginals.flatten(), size=1) / N
            marginals_rand = np.reshape(marginals_rand, [2, 2])
            while not (marginals_rand[1, 1] * marginals_rand[0, 1]):
                marginals_rand = np.random.multinomial(N, true_marginals.flatten(), size=1) / N
                marginals_rand = np.reshape(marginals_rand, [2, 2])
            marginals_rand = np.reshape(marginals_rand, [2, 2])

            data, data_label, data_sensitive = upload_data(ds=ds, n_samples=N, marginals=marginals_rand)
            data_tuple = [data, data_sensitive, data_label]
            c = limiting_dist_EQOPP(X=data, a=data_sensitive, y=data_label, beta=beta, marginals=marginals_rand)
            thetas[rep, i] = c

            end = time.time()
            time_elapsed = (end - start) * (replications - rep - 1)
            conversion = datetime.timedelta(seconds=time_elapsed)
            print('Replication====>' + str(rep) + '/' + str(replications) + ', Time remaining : ' + str(
                conversion))

    # np.savetxt('results_' + str(N_range) + '_iterations_' + str(replications) + '.out', R)
    np.savetxt('limiting_dist_params' + '.out', thetas)


