########################################################################
# A Statistical Test for Probabilistic Fairness
# ACM FACCT 2021
# Authors: Bahar Taskesen, Jose Blanchet, Daniel Kuhn, Viet-Anh Nguyen
########################################################################
from load_data import *
from random import seed, shuffle
from functions import *
import time
import datetime
from scipy.stats import gamma


def sigmoid_func(beta, x):
    return 1 / (1 + np.exp(- beta.T @ x))


if __name__ == "__main__":
    # LOAD DATASET
    ds = 'toy_correlated' ## Name of the dataset
    tol = 1e-10
    true_marginals = np.array(([.4, .1], [.3, .2])) ## np.array(([P_00, P_01], [P_10, P_11]))
    beta = np.array([0, 1])
    range_gamma = [-5, 5]
    range_of_k = np.linspace(0, 1/8, 100)
    replications = 1
    N_range = [100] #[10, 100, 1000, 5000]
    #########
    R = np.zeros([replications, len(N_range)])
    test_res = np.zeros([replications, len(N_range)])
    print('True marginals=' + str(true_marginals))
    print('range of gamma:' + str(range_gamma))
    print('range of k:' + 'from' + str(min(range_of_k)) + ' to ' + str(max(range_of_k)) + ' with ' + str(range_of_k.shape)
          + ' number of steps')
    print('Replications=' + str(replications))
    print('Beta=' + str(beta))
    print('N_range=' + str(N_range))
    print('Tolerance for gamma:' + str(tol))
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

            dist = calculate_distance_prob_eqopp(data_tuple=data_tuple,
                                                 function_of_gamma=f_gamma,
                                                 range_gamma=range_gamma,
                                                 k_opt_fcn=k_opt,
                                                 range_of_k=range_of_k,
                                                 beta=beta, tol=tol)
            R[rep, i] = dist
            c = limiting_dist_EQOPP(X=data, a=data_sensitive, y=data_label, beta=beta, marginals=marginals_rand)

            k = 1 / 2
            theta = 2 * c
            if N * dist > gamma.ppf(.95, a=k, scale=theta):
                print('Reject')
                test_res[rep, i] = 1
            else:
                print('Fail to reject')
                test_res[rep, i] = 0
            end = time.time()
            time_elapsed = (end - start) * (replications - rep - 1)
            conversion = datetime.timedelta(seconds=time_elapsed)
            print('Replication====>' + str(rep) + '/' + str(replications) + ', Time remaining : ' + str(
                conversion))

    np.savetxt('results_' + str(N_range) + '_iterations_' + str(replications) + '.out', R)
    np.savetxt('test_results_' + str(N_range) + '_iterations_' + str(replications) + '.out', test_res)

