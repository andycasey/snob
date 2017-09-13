
"""
Do chemical tagging on a fake data set of open cluster stars.
"""

import numpy as np
import pickle
from astropy.table import Table
from sklearn import (decomposition, mixture)

np.random.seed(42)

catalog = Table.read("data/catalog.fits")
with open("data/generated-parameters.pkl", "rb") as fp:
    gp = pickle.load(fp)


def converged_mixture_model(y, model_class, n_components, K_seq=3, **kwargs):

    # Use BIC as our metric.
    metric = lambda fitted_model: fitted_model.bic(y)

    n_components = np.atleast_1d(n_components).astype(int)

    converged = False
    metrics = np.nan * np.ones(n_components.size, dtype=float)
    
    for i, K in enumerate(n_components):
        try:
            model = model_class(n_components=K, **kwargs)
            fitted = model.fit(y)

        except ValueError:
            print("Failed on {} for K = {}".format(model_class, K))

            # Don't run any more trials, as increasing K will certainly fail.
            break

        else:
            metrics[i] = metric(fitted)

            if K_seq is not None and K > K_seq - 1 \
            and np.all(np.diff(metrics[-K_seq:]) > 0):
                converged = True
                break

    K_best = 1 + np.argmin(metrics)

    return (K_best, converged, metrics)





X = np.vstack([catalog[name] for name in catalog.dtype.names[2:]]).T
K_clusters_total = len(set(catalog["cluster_id"]))



# Use a standard Gaussian mixture model with BIC penalisation.

if False:

    M = 10 # number of monte carlo realisations to do at each K
    results = []

    for K_true in range(1, K_clusters_total):

        for m in range(M):
            if m > 0 and K_true == K_clusters_total: break

            # Pick which clusters we will include,
            selected_cluster_ids = np.random.choice(np.arange(K_clusters_total),
                size=K_true, replace=False)

            match = np.in1d(catalog["cluster_id"], selected_cluster_ids)

            K_trials = np.arange(1, np.max([2 * K_true, 10]))
            K_best, converged, metrics = converged_mixture_model(X[match],
                mixture.GaussianMixture, K_trials)

            results.append((K_true, K_best, converged))

            print(results[-1])


# Use standard FactorAnalysis



M = 10

for K_true in range(10, K_clusters_total):

    for m in range(M):
        if m > 0 and K_true == K_clusters_total: break

        # Pick which clusters we will include.
        selected_cluster_ids = np.random.choice(np.arange(K_clusters_total),
            size=K_true, replace=False)

        match = np.in1d(catalog["cluster_id"], selected_cluster_ids)


        model = decomposition.FactorAnalysis(n_components=1)

        model = model.fit(X[match])

        raise a

raise a