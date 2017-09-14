
"""
Do chemical tagging on a fake data set of open cluster stars.
"""

import numpy as np
import pickle
from astropy.table import Table
from sklearn import (decomposition, mixture)

import utils
from snob import mixture_slf as snob

random_seed = 42

K_start = 1
M = 10 # number of monte carlo realisations to do at each K


catalog = Table.read("data/catalog.fits")
with open("data/generated-parameters.pkl", "rb") as fp:
    gp = pickle.load(fp)

    




X = np.vstack([catalog[name] for name in catalog.dtype.names[2:]]).T
K_clusters_total = len(set(catalog["cluster_id"]))


# Use a standard Gaussian mixture model with BIC penalisation.

if True:

    gmm_kwds = dict()

    np.random.seed(random_seed)

    results = []
    slf_results = []

    running_delta_slf = 0
    running_delta_gmm = 0

    print("Standard GMM with kwds: {}".format(gmm_kwds))

    for K_true in range(K_start, K_clusters_total):

        for m in range(M):
            if m > 0 and K_true == K_clusters_total: break

            # Pick which clusters we will include,
            selected_cluster_ids = np.random.choice(np.arange(K_clusters_total),
                size=K_true, replace=False)

            match = np.in1d(catalog["cluster_id"], selected_cluster_ids)
            K_trials = np.arange(1, np.max([2 * K_true, 10]))

            # Standard GMM
            K_best, converged, metrics = utils.converged_mixture_model(
                X[match], mixture.GaussianMixture, K_trials, **gmm_kwds)

            results.append((K_true, K_best, converged))

            running_delta_gmm += abs(K_true - K_best)

            print("Stantard GMM", results[-1], running_delta_gmm)

            # Now factor analysis.            
            model = decomposition.FactorAnalysis(n_components=1)
            model = model.fit(X[match])

            # Now run a GMM on the transformed X data.
            X_transformed = model.transform(X[match])
            K_best, converged, metrics = utils.converged_mixture_model(
                X_transformed, mixture.GaussianMixture, K_trials)

            slf_results.append((K_true, K_best, converged))
            running_delta_slf += abs(K_true - K_best)

            print("SLF + GMM   ", slf_results[-1], running_delta_slf)



            if K_true > 5:
                raise a

raise a

# Use standard FactorAnalysis


if True:
    print("Clustering on factor scores")

    np.random.seed(random_seed)

    slf_results = []

    for K_true in range(K_start, K_clusters_total):

        for m in range(M):
            if m > 0 and K_true == K_clusters_total: break

            # Pick which clusters we will include.
            selected_cluster_ids = np.random.choice(np.arange(K_clusters_total),
                size=K_true, replace=False)

            match = np.in1d(catalog["cluster_id"], selected_cluster_ids)

            model = decomposition.FactorAnalysis(n_components=1, tol=1e-8,
                max_iter=100000)
            model = model.fit(X[match])

            #utils.compare_latent_factors(model, gp)

            # Now do K-means on ...what 
            X_transformed = model.transform(X[match])

            K_trials = np.arange(1, np.max([2 * K_true, 10]))
            K_best, converged, metrics = utils.converged_mixture_model(
                X_transformed, mixture.GaussianMixture, K_trials)

            slf_results.append((K_true, K_best, converged))

            print(slf_results[-1])


# Use SLF + GMM in MML

raise a
if True:
    print("Doing MML")

    np.random.seed(random_seed)

    slf_results = []




raise a