
"""
Steps:

1. Generate some data

2. Fit those data using K-means++ with K from 1 ... K_true + 10

3. Record the log likelihood improvement for each thing.

"""

import logging
import numpy as np
from sklearn import (cluster, datasets)
from collections import Counter
from snob import mixture_search

np.random.seed(42)


def _generate_data(N=None, D=None, K=None, cluster_std=1.0, 
    center_box=(-10, 10.0), shuffle=True, random_state=None):

    if K is None:
        K = max(1, abs(int(np.random.normal(0, 100))))

    if N is None:
        N = int(np.random.uniform(K, K**2))

    if D is None:
        D = int(np.random.uniform(1, 10))

    kwds = dict(n_samples=N, n_features=D, centers=K,
        cluster_std=cluster_std, center_box=center_box, shuffle=shuffle,
        random_state=random_state)
    X, y = datasets.make_blobs(**kwds)
    return (X, y, kwds)


def _quick_fit_data(y, K, **kwargs):

    model = KMeans(n_clusters=K, **kwargs)
    model.fit(y)

    # calculate LL, etc.

    raise a



experiments = 10

data = {}
for experiment_number in range(experiments):
    data[experiment_number] = _generate_data()


# calculate total size of arrays.
K_trues = [kwds["centers"] for (X, labels, kwds) in data.values()]
max_K = max(K_trues)

print("K_trues: {} {}".format(max_K, K_trues))

raise a

K_trials = np.repeat(np.arange(1, max_K + 1), experiments)
K_trials = K_trials.reshape((-1, experiments)).T
ll_trials = np.nan * np.ones((experiments, max_K))

for e in range(experiments):

    X, labels, kwds = data[e]

    N = X.shape[0]
    K_true = kwds["centers"]

    # Now go from K=1...K_true + 10
    for i, K in enumerate(K_trials[e]):

        if K > K_true + 10: break

        model = cluster.KMeans(n_clusters=K)
        model.fit(X)

        mu = model.cluster_centers_

        # generate repsonsibilities.
        responsibility = np.zeros((K, N))
        responsibility[model.labels_, np.arange(N)] = 1.0

        # estimate covariance matrices.
        cov = mixture_search._estimate_covariance_matrix_full(X, responsibility, mu)

        weight = responsibility.sum(axis=1)/N

        try:
            R, ll = mixture_search.responsibility_matrix(X, mu, cov, weight,
                covariance_type="full", full_output=True)

        except ValueError:
            logging.exception("Failed to calculate log-likelihood")
        
        else:
            ll_trials[e, i] = ll.sum()

        print(e, i, K, K_true)


fig, ax = plt.subplots()
for e in range(experiments):
    ax.scatter(K_trials[e], ll_trials[e])


raise a