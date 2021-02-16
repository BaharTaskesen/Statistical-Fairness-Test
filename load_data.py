import math
import numpy as np
import matplotlib.pyplot as plt  # for plotting stuff
import pandas as pd
from collections import defaultdict
from sklearn import preprocessing
from scipy.stats import multivariate_normal  # generating synthetic data
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

import os, sys

path = os.getcwd()
# print('\nCurrent path:' + path)

DIR_DATA = path + '/datasets/'


def load_compas_data(DIR_DATA=DIR_DATA):
    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count",
                               "c_charge_degree"]  # features to be used for classification
    CONT_VARIABLES = [
        "priors_count"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "two_year_recid"  # the decision variable
    SENSITIVE_ATTRS = ["race"]

    COMPAS_INPUT_FILE = DIR_DATA + "compas/compas-scores-two-years.csv"
    print('Loading COMPAS dataset...')
    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df = df.dropna(subset=["days_b_screening_arrest"])  # dropping missing vals

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Filtering the data """

    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
    idx = np.logical_and(data["days_b_screening_arrest"] <= 30, data["days_b_screening_arrest"] >= -30)

    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O")  # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    # y[y == 0] = -1

    print("\nNumber of people recidivating within two years")
    print(pd.Series(y).value_counts())
    print("\n")
    X = np.array([]).reshape(len(y),
                             0)  # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)

    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in SENSITIVE_ATTRS:
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)
            x_control[attr] = vals
            pass
        else:
            if attr in CONT_VARIABLES:
                vals = [float(v) for v in vals]
                vals = preprocessing.scale(vals)  # 0 mean and 1 variance
                vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col

            else:  # for binary categorical variables, the label binarizer uses just one var instead of two
                lb = preprocessing.LabelBinarizer()
                lb.fit(vals)
                vals = lb.transform(vals)

            # add to sensitive features dict


            # add to learnable features
            X = np.hstack((X, vals))

            if attr in CONT_VARIABLES:  # continuous feature, just append the name
                feature_names.append(attr)
            else:  # categorical features
                if vals.shape[1] == 1:  # binary features that passed through lib binarizer
                    feature_names.append(attr)
                else:
                    for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
                        feature_names.append(attr + "_" + str(k))

    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()
    # sys.exit(1)

    # """permute the date randomly"""
    # perm = range(0, X.shape[0])
    # shuffle(perm)
    # X = X[perm]
    # y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][:]
    # intercept = np.ones(X.shape[0]).reshape(X.shape[0], 1)
    # X = np.concatenate((intercept, X), axis=1)

    assert (len(feature_names) == X.shape[1])
    print("Features we will be using for classification are:", feature_names, "\n")
    x_control = x_control['race']
    return X, y, x_control


def load_drug_data(DIR_DATA=DIR_DATA):
    g = pd.read_csv(DIR_DATA + "/drug/drug_consumption.data.txt", header=None, sep=',')
    # g = pd.read_csv("drug_consumption.data.txt", header=None, sep=',')
    g = np.array(g)
    data = np.array(g[:, 1:13])  # Remove the ID and labels
    labels = g[:, 13:]
    yfalse_value = 'CL0'
    y = np.array([1.0 if yy == yfalse_value else 0.0 for yy in labels[:, 5]])
    dataset = namedtuple('_', 'data, target')(data, y)
    print('Loading Drug (black vs others) dataset...')
    # dataset_train = load_drug()
    sensible_feature = 4  # ethnicity
    a = np.array([1.0 if el == -0.31685 else 0 for el in data[:, sensible_feature]])
    X = np.delete(data, sensible_feature, axis=1).astype(float)
    return X, y, a


def load_arrhythmia(DIR_DATA=DIR_DATA):
    from scipy.stats import mode
    arrhythmia = pd.read_csv(DIR_DATA + "/arrhythmia/arrhythmia.data.txt", header=None)
    arrhythmia = np.where(np.isnan(arrhythmia), mode(arrhythmia, axis=0), arrhythmia)[1]
    y = np.array([1.0 if yy == 1 else 0 for yy in arrhythmia[:, -1]])
    data = arrhythmia[:, :-1]
    sensible_feature = 1  # gender
    print('Load Arrhythmiad dataset...')
    print('Different values of the sensible feature', sensible_feature, ':',
          set(data[:, sensible_feature]))
    X = np.delete(data, sensible_feature, axis=1).astype(float)
    a = data[:, sensible_feature]
    data_red = X[:, :12]
    return data_red, y, a


def generate_synthetic_data_zafar(plot_data=True, n_samples=1200):
    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """

     # generate these many data points per class
    disc_factor = math.pi / 4.0  # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination

    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv, X, y

    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2, -2], [[10, 1], [1, 3]]
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1)  # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1)  # negative class

    # join the posisitve and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = np.random.randint(0, X.shape[0], n_samples * 2)

    X = X[perm]
    y = y[perm]

    rotation_mult = np.array(
        [[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_mult)

    """ Generate the sensitive feature here """
    x_control = []  # this array holds the sensitive feature value
    for i in range(len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)

        # normalize the probabilities from 0 to 1
        s = p1 + p2
        p1 = p1 / s
        p2 = p2 / s

        r = np.random.uniform()  # generate a random number from 0 to 1

        if r < p1:  # the first cluster is the positive class
            x_control.append(1.0)  # 1.0 means its male
        else:
            x_control.append(0.0)  # 0.0 -> female

    x_control = np.array(x_control)

    """ Show the data """
    if plot_data:
        num_to_draw = 200  # we will only draw a small number of points to avoid clutter
        x_draw = X[:num_to_draw]
        y_draw = y[:num_to_draw]
        x_control_draw = x_control[:num_to_draw]

        X_s_0 = x_draw[x_control_draw == 0.0]
        X_s_1 = x_draw[x_control_draw == 1.0]
        y_s_0 = y_draw[x_control_draw == 0.0]
        y_s_1 = y_draw[x_control_draw == 1.0]
        plt.scatter(X_s_0[y_s_0 == 1.0][:, 0], X_s_0[y_s_0 == 1.0][:, 1], color='green', marker='x', s=30,
                    linewidth=1.5, label="Prot. +ve")
        plt.scatter(X_s_0[y_s_0 == -1.0][:, 0], X_s_0[y_s_0 == -1.0][:, 1], color='red', marker='x', s=30,
                    linewidth=1.5, label="Prot. -ve")
        plt.scatter(X_s_1[y_s_1 == 1.0][:, 0], X_s_1[y_s_1 == 1.0][:, 1], color='green', marker='o', facecolors='none',
                    s=30, label="Non-prot. +ve")
        plt.scatter(X_s_1[y_s_1 == -1.0][:, 0], X_s_1[y_s_1 == -1.0][:, 1], color='red', marker='o', facecolors='none',
                    s=30, label="Non-prot. -ve")

        plt.tick_params(axis='x', which='both', bottom='off', top='off',
                        labelbottom='off')  # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc=2, fontsize=15)
        plt.xlim((-15, 10))
        plt.ylim((-10, 15))
        plt.show()

        y[y==-1] = 0


    return X, y, x_control


def load_adult(DIR_DATA=DIR_DATA, smaller=False, scaler=True):
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    data = pd.read_csv(DIR_DATA + "/adult/adult.data",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
            )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(DIR_DATA + "adult/adult.test",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    target = np.array([-1.0 if val == 0 else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if smaller:
        print('A smaller version of the dataset is loaded...')
        data = namedtuple('_', 'data, target')(datamat[:len_train // 20, :-1], target[:len_train // 20])
        data_test = namedtuple('_', 'data, target')(datamat[len_train:, :-1], target[len_train:])
    else:
        print('The dataset is loaded...')
        data = namedtuple('_', 'data, target')(datamat[:len_train, :-1], target[:len_train])
        data_test = namedtuple('_', 'data, target')(datamat[len_train:, :-1], target[len_train:])

    dataset_train, dataset_test = data, data_test
    y_train_all = dataset_train.target
    y_train_all[y_train_all == -1] = 0
    y_test = dataset_test.target
    y_test[y_test == -1] = 0
    sensible_feature = 9  # GENDER
    # sensible_feature = dataset_train.data.shape[1]-1
    sensible_feature_values = sorted(list(set(dataset_train.data[:, sensible_feature])))
    print('Different values of the sensible feature', sensible_feature, ':', sensible_feature_values)
    ntrain = len(dataset_train.target)
    dataset_train_sensitive_free = np.delete(dataset_train.data, sensible_feature, 1)
    sensitive_attributes_train = dataset_train.data[:, sensible_feature]
    sensitive_attributes_train[sensitive_attributes_train == min(sensitive_attributes_train)] = 0
    sensitive_attributes_train[sensitive_attributes_train == max(sensitive_attributes_train)] = 1

    dataset_test_sensitive_free = np.delete(dataset_test.data, sensible_feature, 1)
    X_test = dataset_test_sensitive_free
    test_sensitives = dataset_test.data[:, sensible_feature]
    test_sensitives[test_sensitives == min(test_sensitives)] = 0
    test_sensitives[test_sensitives == max(test_sensitives)] = 1
    a_test = test_sensitives

    X = dataset_train_sensitive_free
    a = sensitive_attributes_train
    y = y_train_all
    return X, y, a, X_test, y_test, a_test


# def load_toy_test():
#     # Load toy test
#     n_samples = 100 * 2
#     n_samples_low = 20 * 2
#     n_dimensions = 10
#     X, y, sensible_feature_id, _, _ = generate_toy_data(n_samples=n_samples,
#                                                         n_samples_low=n_samples_low,
#                                                         n_dimensions=n_dimensions)
#     data = namedtuple('_', 'data, target')(X, y)
#     return data, data


def generate_correlated_data(n_samples, marginals):
    def gen_gaussian_diff_size(mean_in, cov_in, z_val, class_label, n):
        """
        mean_in: mean of the gaussian cluster
        cov_in: covariance matrix
        z_val: sensitive feature value
        class_label: +1 or -1
        n: number of points
        """

        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(n)
        y = np.ones(n, dtype=float) * class_label
        z = np.ones(n, dtype=float) * z_val  # all the points in this cluster get this value of the sensitive attribute

        return nv, X, y, z


    """
    Generate data such that a classifier optimizing for accuracy will have disparate false positive rates as well as disparate false negative rates for both groups.
    """

    # cc = [[3.5, 0], [0, 3.5]]
    # mu1, sigma1 = [6, 0], cc  # z=1, +
    # cc = [[5, 0], [0, 5]]
    # mu2, sigma2 = [-2, 0], cc  # z=0, +
    #
    # cc = [[3.5, 0], [0, 3.5]]
    # mu3, sigma3 = [2, 0], cc  # z=1, -
    # cc = [[5, 0], [0, 5]]
    # mu4, sigma4 = [-6, 0], cc  # z=0, -
    cc = [[3.5, 0], [0, 5]]
    mu1, sigma1 = [6, 0], cc  # z=1, +
    cc = [[5, 0], [0, 5]]
    mu2, sigma2 = [-2, 0], cc  # z=0, +

    cc = [[3.5, 0], [0, 5]]
    mu3, sigma3 = [6, 0], cc  # z=1, -
    cc = [[5, 0], [0, 5]]
    mu4, sigma4 = [-4, 0], cc  # z=0, -

    nv1, X1, y1, z1 = gen_gaussian_diff_size(mu1, sigma1, 1, 1, int(n_samples * marginals[1, 1]))  # z=1, +
    nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, 1, int(n_samples * marginals[0, 1]))  # z=0, +
    nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * marginals[1, 0]))  # z=1, -
    nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * marginals[0, 0]))  # z=0, -


    # merge the clusters
    X = np.vstack((X1, X2, X3, X4))
    y = np.hstack((y1, y2, y3, y4))
    x_control = np.hstack((z1, z2, z3, z4))

    # shuffle the data
    # perm = np.random.randint(low=0, high=int(X.shape[0]), size=int(n_samples))
    # np.random.shuffle(perm)
    # X_shuffled = X[perm]
    # y_shuffled = y[perm]
    # x_control_shuffled = x_control[perm]

    return X, y, x_control



def generate_correlated_data_deneme(n_samples, marginals):
    def gen_gaussian_diff_size(mean_in, cov_in, z_val, class_label, n):
        """
        mean_in: mean of the gaussian cluster
        cov_in: covariance matrix
        z_val: sensitive feature value
        class_label: +1 or -1
        n: number of points
        """

        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(n)
        y = np.ones(n, dtype=float) * class_label
        z = np.ones(n, dtype=float) * z_val  # all the points in this cluster get this value of the sensitive attribute

        return nv, X, y, z


    """
    Generate data such that a classifier optimizing for accuracy will have disparate false positive rates as well as disparate false negative rates for both groups.
    """

    # cc = [[3.5, 0], [0, 3.5]]
    # mu1, sigma1 = [6, 0], cc  # z=1, +
    # cc = [[5, 0], [0, 5]]
    # mu2, sigma2 = [-2, 0], cc  # z=0, +
    #
    # cc = [[3.5, 0], [0, 3.5]]
    # mu3, sigma3 = [2, 0], cc  # z=1, -
    # cc = [[5, 0], [0, 5]]
    # mu4, sigma4 = [-6, 0], cc  # z=0, -
    cc = [[3.5, 0], [0, 3.5]]
    mu1, sigma1 = [6, 0], cc  # z=1, +
    cc = [[3.5, 0], [0, 3.5]]
    mu2, sigma2 = [-2, 0], cc  # z=0, +

    cc = [[3.5, 0], [0, 5]]
    mu3, sigma3 = [6, 0], cc  # z=1, -
    cc = [[5, 0], [0, 5]]
    mu4, sigma4 = [-4, 0], cc  # z=0, -

    nv1, X1, y1, z1 = gen_gaussian_diff_size(mu1, sigma1, 1, 1, int(n_samples * marginals[1, 1]))  # z=1, +
    nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, 1, int(n_samples * marginals[0, 1]))  # z=0, +
    nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * marginals[1, 0]))  # z=1, -
    nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * marginals[0, 0]))  # z=0, -


    # merge the clusters
    X = np.vstack((X1, X2, X3, X4))
    y = np.hstack((y1, y2, y3, y4))
    x_control = np.hstack((z1, z2, z3, z4))

    # shuffle the data
    # perm = np.random.randint(low=0, high=int(X.shape[0]), size=int(n_samples))
    # np.random.shuffle(perm)
    # X_shuffled = X[perm]
    # y_shuffled = y[perm]
    # x_control_shuffled = x_control[perm]

    return X, y, x_control


def generate_correlated_data_for_mfd(n_samples, marginals):
    def gen_gaussian_diff_size(mean_in, cov_in, z_val, class_label, n):
        """
        mean_in: mean of the gaussian cluster
        cov_in: covariance matrix
        z_val: sensitive feature value
        class_label: +1 or -1
        n: number of points
        """

        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(n)
        y = np.ones(n, dtype=float) * class_label
        z = np.ones(n, dtype=float) * z_val  # all the points in this cluster get this value of the sensitive attribute

        return nv, X, y, z


    """
    Generate data such that a classifier optimizing for accuracy will have disparate false positive rates as well as disparate false negative rates for both groups.
    """

    # cc = [[3.5, 0], [0, 3.5]]
    # mu1, sigma1 = [6, 0], cc  # z=1, +
    # cc = [[5, 0], [0, 5]]
    # mu2, sigma2 = [-2, 0], cc  # z=0, +
    #
    # cc = [[3.5, 0], [0, 3.5]]
    # mu3, sigma3 = [2, 0], cc  # z=1, -
    # cc = [[5, 0], [0, 5]]
    # mu4, sigma4 = [-6, 0], cc  # z=0, -
    cc = [[1.2*2, 0], [0, 1.1*2]]
    mu1, sigma1 = [.2, 2], cc  # z=1, +
    cc = [[1.5*2, 0], [0, 1.5*2]]
    mu2, sigma2 = [.5, 1], cc  # z=0, +

    cc = [[1.5, 0], [0, 2.2]]
    mu3, sigma3 = [-.5, -1], cc  # z=1, -
    cc = [[1.5, 0], [0, 1.5]]
    mu4, sigma4 = [-.5, -2], cc  # z=0, -

    nv1, X1, y1, z1 = gen_gaussian_diff_size(mu1, sigma1, 1, 1, int(n_samples * marginals[1, 1]))  # z=1, +
    nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, 1, int(n_samples * marginals[0, 1]))  # z=0, +
    nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * marginals[1, 0]))  # z=1, -
    nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * marginals[0, 0]))  # z=0, -


    # merge the clusters
    X = np.vstack((X1, X2, X3, X4))
    y = np.hstack((y1, y2, y3, y4))
    x_control = np.hstack((z1, z2, z3, z4))

    # shuffle the data
    # perm = np.random.randint(low=0, high=int(X.shape[0]), size=int(n_samples))
    # np.random.shuffle(perm)
    # X_shuffled = X[perm]
    # y_shuffled = y[perm]
    # x_control_shuffled = x_control[perm]

    return X, y, x_control

def generate_correlated_data_for_mfd2(n_samples, marginals):
    def gen_gaussian_diff_size(mean_in, cov_in, z_val, class_label, n):
        """
        mean_in: mean of the gaussian cluster
        cov_in: covariance matrix
        z_val: sensitive feature value
        class_label: +1 or -1
        n: number of points
        """

        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(n)
        y = np.ones(n, dtype=float) * class_label
        z = np.ones(n, dtype=float) * z_val  # all the points in this cluster get this value of the sensitive attribute

        return nv, X, y, z


    """
    Generate data such that a classifier optimizing for accuracy will have disparate false positive rates as well as disparate false negative rates for both groups.
    """

    # cc = [[3.5, 0], [0, 3.5]]
    # mu1, sigma1 = [6, 0], cc  # z=1, +
    # cc = [[5, 0], [0, 5]]
    # mu2, sigma2 = [-2, 0], cc  # z=0, +
    #
    # cc = [[3.5, 0], [0, 3.5]]
    # mu3, sigma3 = [2, 0], cc  # z=1, -
    # cc = [[5, 0], [0, 5]]
    # mu4, sigma4 = [-6, 0], cc  # z=0, -
    cc = [[1.2 * 2, 0], [0, 1.1 * 2]]
    mu1, sigma1 = [-.1, 1.5], cc  # z=1, +
    cc = [[1.5* 2, 0], [0, 1.5* 2]]
    mu2, sigma2 = [-.5, .75], cc  # z=0, +

    cc = [[1.2* 2, 0], [0, 1.1* 2]]
    mu3, sigma3 = [.5, -.75], cc  # z=1, -
    cc = [[1.5* 2, 0], [0, 1.5* 2]]
    mu4, sigma4 = [.1, -1.5], cc  # z=0, -

    nv1, X1, y1, z1 = gen_gaussian_diff_size(mu1, sigma1, 1, 1, int(n_samples * marginals[1, 1]))  # z=1, +
    nv2, X2, y2, z2 = gen_gaussian_diff_size(mu2, sigma2, 0, 1, int(n_samples * marginals[0, 1]))  # z=0, +
    nv3, X3, y3, z3 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * marginals[1, 0]))  # z=1, -
    nv4, X4, y4, z4 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * marginals[0, 0]))  # z=0, -


    # merge the clusters
    X = np.vstack((X1, X2, X3, X4))
    y = np.hstack((y1, y2, y3, y4))
    x_control = np.hstack((z1, z2, z3, z4))

    # shuffle the data
    # perm = np.random.randint(low=0, high=int(X.shape[0]), size=int(n_samples))
    # np.random.shuffle(perm)
    # X_shuffled = X[perm]
    # y_shuffled = y[perm]
    # x_control_shuffled = x_control[perm]

    return X, y, x_control


def generate_correlated_data_array(n_samples, marginals):
    def gen_gaussian_diff_size(mean_in, cov_in, z_val, class_label, n):
        """
        mean_in: mean of the gaussian cluster
        cov_in: covariance matrix
        z_val: sensitive feature value
        class_label: +1 or -1
        n: number of points
        """

        nv = multivariate_normal(mean=mean_in, cov=cov_in)
        X = nv.rvs(n)
        y = np.ones(n, dtype=float) * class_label
        z = np.ones(n, dtype=float) * z_val  # all the points in this cluster get this value of the sensitive attribute

        return nv, X, y, z


    """
    Generate data such that a classifier optimizing for 
    accuracy will have disparate false positive rates
     as well as disparate false negative rates for both groups.
    """

    cc = [[3.5, 0], [0, 5]]
    mu1, sigma1 = [6, 0], cc  # z=1, +
    cc = [[5, 0], [0, 5]]
    mu2, sigma2 = [-2, 0], cc  # z=0, +

    cc = [[3.5, 0], [0, 5]]
    mu3, sigma3 = [6, 0], cc  # z=1, -
    cc = [[5, 0], [0, 5]]
    mu4, sigma4 = [-4, 0], cc  # z=0, -

    nv1, X11, y11, z11 = gen_gaussian_diff_size(mu1, sigma1, 1, 1, int(n_samples * marginals[1, 1]))  # z=1, +
    nv2, X01, y01, z01 = gen_gaussian_diff_size(mu2, sigma2, 0, 1, int(n_samples * marginals[0, 1]))  # z=0, +
    nv3, X10, y10, z10 = gen_gaussian_diff_size(mu3, sigma3, 1, 0, int(n_samples * marginals[1, 0]))  # z=1, -
    nv4, X00, y00, z00 = gen_gaussian_diff_size(mu4, sigma4, 0, 0, int(n_samples * marginals[0, 0]))  # z=0, -


    # merge the clusters
    X = [X00, X01, X10, X11]
    y = [y00, y01, y10, y11]
    a = [z00, z01, z10, z11]
    # shuffle the data
    # perm = np.random.randint(low=0, high=int(X.shape[0]), size=int(n_samples))
    # np.random.shuffle(perm)
    # X_shuffled = X[perm]
    # y_shuffled = y[perm]
    # x_control_shuffled = x_control[perm]

    return X, y, a


def upload_data(ds, n_samples=None, marginals=None):
    if ds == 'adult':
        test_set_fixed = 1
        K3 = 1
        X, y, a, X_test, y_test, a_test = load_adult(DIR_DATA=DIR_DATA)
    elif ds == 'zafar_toy':
        test_set_fixed = 0
        X, y, a = generate_synthetic_data_zafar(plot_data=True)
    elif ds == 'compas':
        test_set_fixed = 0
        X, y, a = load_compas_data(DIR_DATA=DIR_DATA)
        # rhos = np.linspace(0.01, 0.03, num=50)
    elif ds == 'drug':
        test_set_fixed = 0
        X, y, a = load_drug_data(DIR_DATA=DIR_DATA)
    elif ds == 'arrhythmia':
        test_set_fixed = 0
        X, y, a = load_arrhythmia(DIR_DATA=DIR_DATA)
    elif ds == 'toy_correlated':
        X, y, a = generate_correlated_data(n_samples=n_samples, marginals=marginals)
    elif ds == 'toy_correlated_deneme':
        X, y, a = generate_correlated_data_deneme(n_samples=n_samples, marginals=marginals)
    elif ds == 'toy_correlated_array':
        X, y, a = generate_correlated_data_array(n_samples=n_samples, marginals=marginals)
    elif ds == 'toy_correlated_for_most_fav_dist':
        X, y, a = generate_correlated_data_for_mfd(n_samples=n_samples, marginals=marginals)
    elif ds == 'toy_correlated_for_most_fav_dist2':
        X, y, a = generate_correlated_data_for_mfd2(n_samples=n_samples, marginals=marginals)
    else:
        X, y, a = None, None, None
        print('Please enter a correct data type...')
    return X, y, a