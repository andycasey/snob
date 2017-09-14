import numpy as np
import matplotlib.pyplot as plt

from time import time



def converged_mixture_model(y, model_class, n_components, K_seq=3, 
    metric_function=None, break_when_breaking=True, **kwargs):
    """
    Run a series of Gaussian mixture models with an increasing number of K
    components.
    """

    if metric_function is None:
        # Use BIC as our metric.
        metric_function = lambda fitted_model: fitted_model.aic(y)


    n_components = np.atleast_1d(n_components).astype(int)

    converged = False
    metrics = np.nan * np.ones(n_components.size, dtype=float)
    
    t_init = time()

    for i, K in enumerate(n_components):
        try:
            model = model_class(n_components=K, **kwargs)
            fitted = model.fit(y)

        except ValueError:
            print("Failed on {} for K = {}".format(model_class, K))

            # Don't run any more trials, as increasing K will certainly fail.
            if break_when_breaking: break
            else: continue

        else:
            metrics[i] = metric_function(fitted)


            #print(i, K, K_seq, K > K_seq - 1, metrics[:i], np.diff(metrics[i - K_seq:i + 1]))

            if K_seq is not None and K > K_seq \
            and np.all(np.diff(metrics[i - K_seq:i + 1]) > 0):
                converged = True
                break

    K_best = 1 + np.nanargmin(metrics)

    meta = dict(time_taken=time() - t_init)

    return (K_best, converged, metrics)




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