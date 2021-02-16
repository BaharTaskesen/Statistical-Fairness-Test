########################################################################
# A Statistical Test for Probabilistic Fairness
# ACM FACCT 2021
# Authors: Bahar Taskesen, Jose Blanchet, Daniel Kuhn, Viet-Anh Nguyen
########################################################################
from load_data import *
from matplotlib import colors as mcolors
from random import seed
from functions import *
from LogisticRegression import LogisticRegression
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def sigmoid_func(beta, x):
    return 1 / (1 + np.exp(- beta.T @ x))


if __name__ == "__main__":
    # LOAD DATASET
    ds = 'toy_correlated_for_most_fav_dist'
    # n_data = 1e6
    tol = 1e-10
    true_marginals = np.array(([.25, .25], [.25, .25])) ## np.array(([P_00, P_01], [P_10, P_11]))
    # beta = np.array([0, 1])
    range_gamma = [-5, 5]
    range_of_k = np.linspace(0, 1/8, 100)
    n_train = 1000 #[10, 100, 1000, 5000]
    n_test = 30
    #########
    SEED = 11223
    seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(2)
    print('True marginals=' + str(true_marginals))
    print('range of gamma:' + str(range_gamma))
    print('range of k:' + 'from' + str(min(range_of_k)) + ' to ' + str(max(range_of_k)) + ' with ' + str(range_of_k.shape)
          + ' number of steps')
    print('Tolerance for gamma:' + str(tol))
    print('SEED:' + str(SEED))

    # marginals_rand = np.random.multinomial(N, true_marginals.flatten(), size=1) / N
    # marginals_rand = np.reshape(marginals_rand, [2, 2])
    # while not (marginals_rand[1, 1] * marginals_rand[0, 1]):
    #     marginals_rand = np.random.multinomial(N, true_marginals.flatten(), size=1) / N
    #     marginals_rand = np.reshape(marginals_rand, [2, 2])
    # marginals_rand = np.reshape(marginals_rand, [2, 2])

    X_train, y_train, a_train = upload_data(ds=ds, n_samples=n_train, marginals=true_marginals)
    X_test, y_test, a_test = upload_data(ds=ds, n_samples=n_test, marginals=true_marginals)
    clf = LogisticRegression(fit_intercept=False, reg=0)
    clf.fit(X_train, y_train)

    # if ds == 'toy_correlated_for_most_fav_dist':
    #     clf.coef_ = np.array([0.25, 0.80])
    # else:
    #     clf.coef_ = np.array([-.25, .8])

    data_tuple = [X_test, a_test, y_test]
    clf.coef_ = np.array([0.4, 1.12])

    dist, optimum_ks, gamma_opt = calculate_distance_prob_eqopp_with_opt_params(data_tuple=data_tuple,
                                         function_of_gamma=f_gamma,
                                         range_gamma=range_gamma,
                                         k_opt_fcn=k_opt,
                                         range_of_k=range_of_k,
                                         beta=clf.coef_, tol=tol)
    h = .02
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # just plot the dataset first
    cm = plt.cm.RdBu
    from matplotlib.colors import ListedColormap
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    from matplotlib import rcParams
    # clf.coef_ = np.array([0.2, 0.8])
    score = clf.score(X_test, y_test)

    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    Z = Z.reshape(xx.shape)
    plt.figure()
    cs = plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    plt.axis('off')
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Times']

    predict = clf.predict(X_test)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    ax = plt.gca()

    rhos = np.linspace(0, 0.2, 1000)

    # use all available CPUs
    weights = []
    x_projs = []
    for i, x_hat in enumerate(X_test):
        if y_test[i] == 1:
            weights.append(optimum_ks[i] * y_test[i] * gamma_opt * (1 / true_marginals[1, 1] * a_test[i] -
                                                                            1 / true_marginals[0, 1] * (1 - a_test[i])))
            x_projs.append(x_hat - y_test[i] * weights[i] * clf.coef_)
        else:
            weights.append(0)
            x_projs.append(x_hat)
    x_projs = np.stack(x_projs)
    cc = ['r', 'b']
    for j in range(X_test.shape[0]):
        if a_test[j] == 1:
            if np.abs(weights[j]) != 0:
                plt.scatter(X_test[j, 0], X_test[j, 1], cmap=cm_bright, marker='o',
                            c=cc[int(y_test[j])], edgecolors='k', label='A=1', s=150, alpha=0.5)
                plt.scatter(x_projs[j, 0], x_projs[j, 1], cmap=cm_bright, marker='o',
                            c=cc[int(y_test[j])], edgecolors='k', label='A=1', s=150, alpha=1)
                # ax.annotate("", xytext=(x_projs[j, 0], x_projs[j, 1]),
                #             xy=(X_test[j, 0], X_test[j, 1]),
                #             arrowprops=dict(arrowstyle="->", lw=3, color=colors['lime']))
                plt.plot([x_projs[j, 0], X_test[j, 0]],
                         [x_projs[j, 1], X_test[j, 1]], c=colors['lime'], linewidth=4, alpha=0.8)
            elif weights[j] == 0:
                # plt.scatter(x_projs[j, 0], x_projs[j, 1], cmap=cm_bright, marker='o',
                #             c=cc[int(y_test[j])], edgecolors='k', label='A=1', s=150, alpha=1)
                plt.scatter(X_test[j, 0], X_test[j, 1], cmap=cm_bright, marker='o',
                            c=cc[int(y_test[j])], edgecolors='k', label='A=1', s=150, alpha=1)
        else:
            if np.abs(weights[j]) != 0:
                plt.scatter(X_test[j, 0], X_test[j, 1],
                            cmap=cm_bright, marker='P', s=150,
                            c=cc[int(y_test[j])], edgecolors='k', label='A=0', alpha=0.5)
                plt.scatter(x_projs[j, 0], x_projs[j, 1],
                            cmap=cm_bright, marker='P', s=150,
                            c=cc[int(y_test[j])], edgecolors='k', label='A=0', alpha=1)
                plt.plot([x_projs[j, 0], X_test[j, 0]],
                         [x_projs[j, 1], X_test[j, 1]], c=colors['lime'],
                         linewidth=4, alpha=0.8)
                # ax.annotate("", xytext=(x_projs[j, 0], x_projs[j, 1]),
                #             xy=(X_test[j, 0], X_test[j, 1]),
                #             arrowprops=dict(arrowstyle="->", lw=3, color=colors['lime']))


            elif weights[j] == 0:
                plt.scatter(X_test[j, 0], X_test[j, 1],
                            cmap=cm_bright, marker='P', s=150,
                            c=cc[int(y_test[j])], edgecolors='k', label='A=0', alpha=1)

        # plt.text(worst_case_support[j, 0] , worst_case_support[j, 1],
        #         ('%d' % score_num).lstrip('0'),
        #         size=10, horizontalalignment='right')
    plt.axis('off')
    plt.xlim(xmin=min(X_test[:, 0]) - .5,
             xmax=max(X_test[:, 0]) + .5)
    plt.ylim(ymin=min(X_test[:, 1]) - .5,
             ymax=max(X_test[:, 1]) + .5)
    # plt.xlim(xmin=-4, xmax=3)
    # plt.ylim(ymin=-5.2, ymax=5)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend(fontsize=13)
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Times']
    if ds == 'toy_correlated_for_most_fav_dist':
        ax.annotate("", xytext=(-3-.4-.5, 1.3), xy=(clf.coef_[0] /LA.norm(clf.coef_) + -3 -.4-.5, clf.coef_[1] /LA.norm(clf.coef_)+ 1.3),
                    arrowprops = dict(arrowstyle="->", lw=3))
    else:
        ax.annotate("", xytext=(-3, -0.7), xy=(clf.coef_[0] /LA.norm(clf.coef_) + -3, clf.coef_[1] /LA.norm(clf.coef_)-.7),
                    arrowprops = dict(arrowstyle="->", lw=3))
    # plt.title("Worst Case Support")
    # plt.text(xx.max()-0.3, yy.min()+.3, ('$\\rho$=%.2f' % rho).lstrip('0'),
    #         horizontalalignment='right', fontsize=13)
    # plt.show()
    plt.tight_layout()
    plt.savefig('most_fav_distribution' + ds + '.pdf', bbox_inches = 'tight', pad_inches = 0)
    # plt.show()