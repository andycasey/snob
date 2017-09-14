
"""
Do chemical tagging on a fake data set of open cluster stars.
"""

import numpy as np
import pickle
from astropy.table import Table
from sklearn import (decomposition, mixture)

random_seed = 42

catalog = Table.read("data/catalog.fits")
with open("data/generated-parameters.pkl", "rb") as fp:
    gp = pickle.load(fp)


K_start = 20

def converged_mixture_model(y, model_class, n_components, K_seq=3, 
    metric_function=None, **kwargs):

    if metric_function is None:
        # Use BIC as our metric.
        metric_function = lambda fitted_model: fitted_model.aic(y)


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
            metrics[i] = metric_function(fitted)

            if K_seq is not None and K > K_seq - 1 \
            and np.all(np.diff(metrics[-K_seq:]) > 0):
                converged = True
                break

    K_best = 1 + np.argmin(metrics)

    return (K_best, converged, metrics)





X = np.vstack([catalog[name] for name in catalog.dtype.names[2:]]).T
K_clusters_total = len(set(catalog["cluster_id"]))

K_clusters_total = K_start + 1

# Use a standard Gaussian mixture model with BIC penalisation.

if True:
    print("Standard GMM")

    np.random.seed(random_seed)

    M = 10 # number of monte carlo realisations to do at each K
    results = []

    for K_true in range(K_start, K_clusters_total):

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


def compare_latent_factors(model, expected_parameters, rtol=1e-3, atol=1e-2):

    # Check orientation.
    signs = np.sign(model.components_/gp["factor_loads"])[0]

    assert len(set(signs)) == 1


    """

    orientation = [-1, +1][signs[0] > 0]

    # TODO: this will fail for multiple latent factors -- vectorize!!
    assert np.allclose(
        model.components_[0], orientation * expected_parameters["factor_loads"],
        rtol=rtol, atol=atol)

    assert np.allclose(model.mean_, expected_parameters["means"],
        rtol=rtol, atol=atol)

    assert np.allclose(
        model.transform(X[match]).flatten(),
        orientation * expected_parameters["factor_scores"][match].flatten(),
        rtol=rtol, atol=atol)
    """

    def _common_limits(ax):
        limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
        limits = (min(limits), max(limits))
        ax.set_xlim(limits)
        ax.set_ylim(limits)


    fig, ax = plt.subplots()
    ax.scatter(model.mean_, expected_parameters["means"])
    ax.set_title("means")
    _common_limits(ax)


    fig, ax = plt.subplots()
    ax.scatter(model.components_[0], expected_parameters["factor_loads"][0])
    ax.set_title("factor loads")
    _common_limits(ax)

    fig, ax = plt.subplots()
    ax.scatter(
        expected_parameters["factor_scores"][match].flatten(),
        model.transform(X[match]).flatten())
    ax.set_title("factor scores")
    _common_limits(ax)


    return True


M = 10

np.random.seed(random_seed)

print("Clustering on factor scores")

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

        compare_latent_factors(model, gp)

        # Now do K-means on ...what 
        X_transformed = model.transform(X[match])

        K_trials = np.arange(1, np.max([2 * K_true, 10]))
        K_best, converged, metrics = converged_mixture_model(X_transformed,
            mixture.GaussianMixture, K_trials)

        slf_results.append((K_true, K_best, converged))

        print(slf_results[-1])



        raise a

raise a