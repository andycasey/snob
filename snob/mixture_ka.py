
"""
An estimator for a mixture of Gaussians, using minimum message length.

The MML formalism is that of (Figueriedo & Jain, 2002), but the search
strategy and score function adopted is that of Kasarapu & Allison (2015).
"""

__all__ = ["GaussianMixtureEstimator", "kullback_leibler_for_multivariate_normals"]

import logging
import numpy as np
import scipy
from collections import defaultdict

logger = logging.getLogger(__name__)

from . import estimator


def _component_covariance_parameters(D, covariance_type):
    r"""
    Return the number of parameters per Gaussian component, given the number 
    of observed dimensions and the covariance type.

    :param D:
        The number of dimensions per data point.

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components.
    """

    if covariance_type == "free":
        return int(D + D*(D + 1)/2.0)
    elif covariance_type == "diag":
        return 2 * D
    elif covariance_type == "tied":
        return D
    elif covariance_type == "tied_diag":
        return D
    else:
        raise ValueError("unknown covariance type '{}'".format(covariance_type))


def _responsibility_matrix(y, mu, cov, weight, small=1e-16):
    r"""
    Return the responsibility matrix,

    .. math::

        r_{ij} = \frac{w_{j}f\left(x_i;\theta_j\right)}{\sum_{k=1}^{K}w_k}f\left(x_i;\theta_k\right)}

    where :math:`r_{ij}` denotes the conditional probability of a datum
    :math:`x_i` belonging to the :math:`j`th component. The effective
    membership associated with each component is then given by

    .. math::

        n_j = \sum_{i=1}^{N}r_{ij} & \textrm{and} & \sum_{j=1}^{K}n_{j} = N

    :param y:
        The data values, :math:`y`.

    :param mu:
        The mean values of the :math:`K` multivariate normal distributions.

    :param cov:
        The covariance matrices of the :math:`K` multivariate normal
        distributions.

    :param weight:
        The current estimates of the relative mixing weight.

    :returns:
        The unnormalized responsibility matrix (the numerator in the equation
        above), and the normalization constants (the denominator in the
        equation above).


    """

    N, D = y.shape
    K = mu.shape[0]

    scalar = (2 * np.pi)**(-D/2.0)
    numerator = np.zeros((K, N))
    for k, (mu_k, cov_k) in enumerate(zip(mu, cov)):

        U, S, V = np.linalg.svd(cov_k)
        Cinv = np.dot(np.dot(V.T, np.linalg.inv(np.diag(S))), U.T)

        O = y - mu_k
        numerator[k] \
            = scalar * weight[k] * np.linalg.det(cov_k)**(-0.5) \
                     * np.exp(-0.5 * np.sum(O.T * np.dot(Cinv, O.T), axis=0))

    denominator = np.clip(np.sum(numerator, axis=0), small, np.inf)
    return (numerator, denominator)

    
def _kill_component(y, mu, cov, weight, component_index):
    r"""
    Kill off a component, and return the new estimates of the component
    parameters.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current estimates of the relative mixing weight.

    :param component_index:
        The index of the component to be killed off.

    :returns:
        A four-length tuple containing the means of the components,
        the covariance matrices of the components, the relative weights of the
        components, and the responsibility matrix.
    """

    mu = np.delete(mu, component_index, axis=0)
    cov = np.delete(cov, component_index, axis=0)
    weight = np.delete(weight, component_index, axis=0)

    K, N = (mu.shape[0], y.shape[0])

    # Normalize the weights.
    weight = weight/np.sum(weight)

    # Calculate responsibilities.
    responsibility = _evaluate_responsibilities(y, mu, cov)

    return (mu, cov, weight, responsibility)


def _initialize(y):
    r"""
    Return initial estimates of the parameters.

    :param y:
        The data values, :math:`y`.

    :param scalar: [optional]
        The scalar to apply (multiplicatively) to the mean of the
        variances along each dimension of the data at initialization time.

    :returns:
        A three-length tuple containing the initial (multivariate) mean,
        the covariance matrix, and the relative weight.
    """

    weight = np.ones((1, 1))

    N, D = y.shape
    mu = np.mean(y, axis=0).reshape((1, -1))
    cov = np.cov(y.T).reshape((1, D, D))
    
    return (mu, cov, weight)



def _expectation(y, mu, cov, weight, N_covpars):
    r"""
    Perform the expectation step of the expectation-maximization algorithm.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The current best estimates of the (multivariate) means of the :math:`K`
        components.

    :param cov:
        The current best estimates of the covariance matrices of the :math:`K`
        components.

    :param weight:
        The current best estimates of the relative weight of all :math:`K`
        components.

    :param N_covpars:
        The number of parameters due to the covariance structure.

    :returns:
        A three-length tuple containing the responsibility matrix,
        the log likelihood, and the change in message length.
    """

    #R = _evaluate_responsibilities(y, mu, cov)
    numerator, denominator = _responsibility_matrix(y, mu, cov, weight)
    responsibility = np.clip(numerator/denominator, 1e-16, 1.0)
    assert np.all(np.isfinite(responsibility))

    # Eq. 40 omitting -Nd\log\eps
    log_likelihood = np.sum(np.log(denominator)) 

    # TODO: check delta_length.
    N, D = y.shape
    K = weight.size
    delta_length = -log_likelihood \
        + (N_covpars/2.0 * np.sum(np.log(weight))) \
        + (N_covpars/2.0 + 0.5) * K * np.log(N)

    # I(K) = K\log{2} + constant

    # Eq. 38
    # I(w) = (M-1)/2 * log(N) - 0.5\sum_{k=1}^{K}\log{w_k} - (K - 1)!

    return (responsibility, log_likelihood, delta_length)



def _maximize_child_components(y, mu, cov, weight, responsibility, 
    parent_responsibility, covariance_type="free", regularization=0, 
    N_covpars=None):
    r"""
    Perform the maximization step of the expectation-maximization algorithm on
    two child components.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current best estimates of the relative weight of all :math:`K`
        components.

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param parent_responsibility:
        An array of length :math:`N` giving the parent component 
        responsibilities.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: `free`).

    :param regularization: [optional]
        A regularizing factor to apply to covariance matrices. In very small
        samples, a regularization factor may be needed to apply to the
        diagonal of the covariance matrices (default: `0`).

    :param N_covpars: [optional]
        The number of parameters due to the covariance structure. If `None` is
        given, this will be calculated, but it's (slightly) faster to include
        this value.
    """

    M = weight.size # Should be 2, but let's allow for bigger splits.
    N, D = y.shape
    N_covpars = N_covpars or _component_covariance_parameters(D, covariance_type)

    # Update the weights.
    effective_membership = np.sum(responsibility, axis=1)
    new_weight = (effective_membership + 0.5)/(N + M/2.0)

    w_responsibility = parent_responsibility * responsibility
    w_effective_membership = np.sum(w_responsibility, axis=1)

    new_mu = np.zeros_like(mu)
    new_cov = np.zeros_like(cov)
    for m in range(M):
        new_mu[m] = np.sum(w_responsibility[m] * y.T, axis=1) \
                  / w_effective_membership[m]

        offset = y - new_mu[m]
        new_cov[m] = np.dot(w_responsibility[m] * offset.T, offset) \
                   / (w_effective_membership[m] - 1)

    return (new_mu, new_cov, new_weight)

    


def _maximization(y, mu, cov, weight, responsibility, covariance_type="free", 
    regularization=0, N_covpars=None):
    r"""
    Perform the maximization step of the expectation-maximization algorithm.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current best estimates of the relative weight of all :math:`K`
        components.

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: `free`).

    :param regularization: [optional]
        A regularizing factor to apply to covariance matrices. In very small
        samples, a regularization factor may be needed to apply to the
        diagonal of the covariance matrices (default: `0`).

    :param N_covpars: [optional]
        The number of parameters due to the covariance structure. If `None` is
        given, this will be calculated, but it's (slightly) faster to include
        this value.

    :returns:
        A four-length tuple containing: a new estimate of the component means,
        the component covariance matrices, the relative weight of each
        component, and the responsibility matrix.
    """

    M = weight.size
    N, D = y.shape
    N_covpars = N_covpars or _component_covariance_parameters(D, covariance_type)

    # Update the weights.
    effective_membership = np.sum(responsibility, axis=1)
    new_weight = (effective_membership + 0.5)/(N + M/2.0)

    # Update means and covariance matrices.
    new_mu = np.zeros_like(mu)
    new_cov = np.zeros_like(cov)
    for m in range(M):
        new_mu[m] = np.sum(responsibility[m] * y.T, axis=1) \
              / effective_membership[m]

        offset = y - new_mu[m]
        new_cov[m] = np.dot(responsibility[m] * offset.T, offset) \
                   / (effective_membership[m] - 1)
        assert np.all(np.isfinite(new_mu[m]))
        assert np.all(np.isfinite(new_cov[m]))

    # Ensure that the weights are normalized.
    new_weight /= np.sum(new_weight)

    return (new_mu, new_cov, new_weight)


def _split_component(y, mu, cov, weight, responsibility, index, 
    covariance_type="free", regularization=0, N_covpars=None, threshold=1e-5,
    **kwargs):
    """
    Split a component from the current mixture and determine the new optimal
    state.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current estimates of the relative mixing weight.

    :param index:
        The index of the component to be split.

    :param N_covpars:
        The number of parameters due to the covariance structure.

    # TODO: other docs.
    # TODO: returns?
    """

    M = weight.size
    N, D = y.shape
    N_covpars = N_covpars or _component_covariance_parameters(D, covariance_type)
    max_iterations = kwargs.get("max_sub_iterations", 10000)

    # Compute the direction of maximum variance of the parent component, and
    # locate two points which are one standard deviation away on either side.
    U, S, V = np.linalg.svd(cov[index])
    child_mu = mu[index] + np.vstack([+V[0], -V[0]]) * np.diag(cov[index])**0.5
    
    # Responsibilities are initialized by allocating the data points to the 
    # closest of the two means.
    child_responsibility = np.vstack([
        np.sum(np.abs(y - child_mu[0]), axis=1),
        np.sum(np.abs(y - child_mu[1]), axis=1)
    ])
    child_responsibility /= np.sum(child_responsibility, axis=0)

    # Calculate the child covariance matrices.
    child_cov = np.zeros((2, D, D))
    child_effective_membership = np.sum(child_responsibility, axis=1)

    for k in (0, 1):
        offset = y - child_mu[k]
        child_cov[k] = np.dot(child_responsibility[k] * offset.T, offset) \
                     / (child_effective_membership[k] - 1)

    child_weight = child_effective_membership.T/child_effective_membership.sum()

    # We will need these later.responsibility
    parent_weight = weight[index]
    parent_responsibility = np.clip(responsibility[index], 1e-16, np.inf)

    # Calculate the initial log-likelihood
    # (don't update child responsibilities)
    _, ll, dl = _expectation(y, child_mu, child_cov, child_weight, N_covpars)

    iterations = 1
    ll_dl = [(ll, dl)]

    while True:


        # Run the maximization step on the child components.
        child_mu, child_cov, child_weight = _maximize_child_components(
            y, child_mu, child_cov, child_weight, child_responsibility,
            parent_responsibility, covariance_type, regularization, N_covpars)

        # Run the expectation step on the updated child components.
        child_responsibility, ll, dl = _expectation(
            y, child_mu, child_cov, child_weight, N_covpars)
        
        # Check for convergence.
        prev_ll, prev_dl = ll_dl[-1]
        relative_delta_ll = (ll - prev_ll)/prev_ll

        # Book-keeping.
        iterations += 1
        ll_dl.append([ll, dl])

        print(iterations, child_weight, relative_delta_ll, ll, dl)
        
        assert np.isfinite(relative_delta_ll)
        
        if np.abs(relative_delta_ll) <= threshold \
        or iterations >= max_iterations:
            break

    meta = dict(warnflag=iterations >=max_iterations)
    if meta["warnflag"]:
        logger.warn("Maximum number of E-M iterations reached ({}) "\
                    "when splitting component index {}".format(
                        max_iterations, component_index))

    # After the chld mixture is locally optimized, we need to integrate it
    # with the untouched M - 1 components to result in a M + 1 component
    # mixture M'.

    # An E-M is finally carried out on the combined M + 1 components to
    # estimate the parameters of M' and result in an optimized 
    # (M + 1)-component mixture.

    # Update the component weights.
    # Note that the child A mixture will remain in index `index`, and the
    # child B mixture will be appended to the end.

    if M > 1:

        weight = np.hstack([weight, [parent_weight]])
        weight[index] = parent_weight * child_weight[0]
        weight[-1]    = parent_weight * child_weight[1]

        # Update the responsibility matrix.
        responsibility = np.vstack([responsibility, [parent_responsibility]])
        responsibility[index] = parent_responsibility * child_responsibility[0]
        responsibility[-1]    = parent_responsibility * child_responsibility[1]

        mu = np.vstack([mu, [mu[index]]])
        mu[index] = child_mu[0]
        mu[-1] = child_mu[1]

        cov = np.vstack([cov, [cov[index]]])
        cov[index] = child_cov[0]
        cov[-1] = child_cov[1]

        # Run expectation-maximization on the M + 1 component mixture.
        while True:

            # Run the maximization step.
            mu, cov, weight = _maximization(y, mu, cov, weight, responsibility, 
                covariance_type, regularization, N_covpars)

            # Run the expectation step.
            responsibility, ll, dl = _expectation(y, mu, cov, weight, N_covpars)

            # Check for convergence.
            prev_ll, prev_dl = ll_dl[-1]
            relative_delta_ll = np.abs((ll - prev_ll)/prev_ll)

            ll_dl.append([ll, dl])

            if relative_delta_ll <= threshold:
                break

    else:
        # Simple case where we don't have to re-run E-M because there was only
        # one component to split.
        mu = child_mu
        cov = child_cov
        weight = child_weight
        responsibility = child_responsibility

    meta["log-likelihood"] = ll

    return (mu, cov, weight, responsibility, meta, dl)


def _delete_component(y, mu, cov, weight, responsibility, index, 
    covariance_type="free", regularization=0, N_covpars=None, threshold=1e-5,
    **kwargs):
    """
    Delete a component from the mixture.
    """

    # Initialize.
    M = weight.size
    N, D = y.shape
    N_covpars = N_covpars or _component_covariance_parameters(D, covariance_type)
    max_iterations = kwargs.get("max_sub_iterations", 10000)

    # Create new component weights.
    parent_weight = weight[index]
    parent_responsibility = responsibility[index]
    
    # Eq. 54-55
    perturbed_weight = np.delete(weight, index) / (1 - parent_weight)
    perturbed_responsibility = np.delete(responsibility, index, axis=0) \
                             / (1 - parent_responsibility)
    perturbed_responsibility = np.clip(perturbed_responsibility, 0, 1)

    perturbed_mu = np.delete(mu, index, axis=0)
    perturbed_cov = np.delete(cov, index, axis=0)

    # Calculate the current log-likelihood.
    _, ll, dl = _expectation(
        y, perturbed_mu, perturbed_cov, perturbed_weight, N_covpars)

    iterations = 1
    ll_dl = [(ll, dl)]

    while True:

        # Perform the maximization step.
        perturbed_mu, perturbed_cov, perturbed_weight = _maximization(
            y, perturbed_mu, perturbed_cov, perturbed_weight,
            perturbed_responsibility, covariance_type, regularization,
            N_covpars)
        assert np.all(np.isfinite(perturbed_cov))

        # Run the expectation step.
        perturbed_responsibility, ll, dl = _expectation(
            y, perturbed_mu, perturbed_cov, perturbed_weight, N_covpars)

        # Check for convergence.
        prev_ll, prev_dl = ll_dl[-1]
        relative_delta_ll = np.abs((ll - prev_ll)/prev_ll)

        ll_dl.append([ll, dl])
        iterations += 1

        if relative_delta_ll <= threshold \
        or iterations >= max_iterations:
            break

    meta = dict(warnflag=iterations >=max_iterations)
    if meta["warnflag"]:
        logger.warn("Maximum number of E-M iterations reached ({}) "\
                    "when deleting component index {}".format(
                        max_iterations, index))


    meta["log-likelihood"] = ll

    return (perturbed_mu, perturbed_cov, perturbed_weight, 
        perturbed_responsibility, meta, dl)


def kullback_leibler_for_multivariate_normals(mu_a, cov_a, mu_b, cov_b):
    r"""
    Return the Kullback-Leibler distance from one multivariate normal
    distribution with mean :math:`\mu_a` and covariance :math:`\Sigma_a`,
    to another multivariate normal distribution with mean :math:`\mu_b` and 
    covariance matrix :math:`\Sigma_b`. The two distributions are assumed to 
    have the same number of dimensions, such that the Kullback-Leibler 
    distance is

    .. math::

        D_{\mathrm{KL}}\left(\mathcal{N}_{a}||\mathcal{N}_{b}\right) = 
            \frac{1}{2}\left(\mathrm{Tr}\left(\Sigma_{b}^{-1}\Sigma_{a}\right) + \left(\mu_{b}-\mu_{a}\right)^\top\Sigma_{b}^{-1}\left(\mu_{b} - \mu_{a}\right) - k + \ln{\left(\frac{\det{\Sigma_{b}}}{\det{\Sigma_{a}}}\right)}\right)


    where :math:`k` is the number of dimensions and the resulting distance is 
    given in units of nats.

    .. warning::

        It is important to remember that 
        :math:`D_{\mathrm{KL}}\left(\mathcal{N}_{a}||\mathcal{N}_{b}\right) \neq D_{\mathrm{KL}}\left(\mathcal{N}_{b}||\mathcal{N}_{a}\right)`.


    :param mu_a:
        The mean of the first multivariate normal distribution.

    :param cov_a:
        The covariance matrix of the first multivariate normal distribution.

    :param mu_b:
        The mean of the second multivariate normal distribution.

    :param cov_b:
        The covariance matrix of the second multivariate normal distribution.
    
    :returns:
        The Kullback-Leibler distance from distribution :math:`a` to :math:`b`
        in units of nats. Dividing the result by :math:`\log_{e}2` will give
        the distance in units of bits.
    """

    U, S, V = np.linalg.svd(cov_a)
    Ca_inv = np.dot(np.dot(V.T, np.linalg.inv(np.diag(S))), U.T)

    U, S, V = np.linalg.svd(cov_b)
    Cb_inv = np.dot(np.dot(V.T, np.linalg.inv(np.diag(S))), U.T)

    k = mu_a.size

    offset = mu_b - mu_a
    return 0.5 * np.sum([
          np.trace(np.dot(Ca_inv, cov_b)),
        + np.dot(offset.T, np.dot(Cb_inv, offset)),
        - k,
        + np.log(np.linalg.det(cov_b)/np.linalg.det(cov_a))
    ])


def _merge_component(y, mu, cov, weight, responsibility, index, 
    covariance_type="free", regularization=0, N_covpars=None, threshold=1e-5,
    **kwargs):
    """
    Merge a component from the mixture with its "closest" component, as
    judged by the Kullback-Leibler distance.
    """

    # Initialize.
    M = weight.size
    N, D = y.shape
    N_covpars = N_covpars or _component_covariance_parameters(D, covariance_type)
    max_iterations = kwargs.get("max_sub_iterations", 10000)

    # Calculate the KL distance to the other distributions.
    D_kl = np.inf * np.ones(weight.size)
    for m in range(M):
        if m == index: continue
        D_kl[m] = kullback_leibler_for_multivariate_normals(
            mu[index], cov[index], mu[m], cov[m])

    a_index = index
    b_index = np.nanargmin(D_kl)

    # Initialize.
    perturbed_weight = np.sum(weight[[a_index, b_index]])
    perturbed_responsibility = np.sum(responsibility[[a_index, b_index]], axis=0)
    effective_membership_k = np.sum(perturbed_responsibility)

    perturbed_mu_k = np.sum(perturbed_responsibility * y.T, axis=1) \
                   / effective_membership_k

    offset = y - perturbed_mu_k
    perturbed_cov = np.dot(perturbed_responsibility * offset.T, offset) \
                  / (effective_membership_k - 1)

    # Delete the b-th component.
    del_index = np.max([a_index, b_index])
    keep_index = np.min([a_index, b_index])
    mu = np.delete(mu, del_index, axis=0)
    cov = np.delete(cov, del_index, axis=0)
    weight = np.delete(weight, del_index, axis=0)
    responsibility = np.delete(responsibility, del_index, axis=0)

    mu[keep_index] = perturbed_mu_k
    cov[keep_index] = perturbed_cov
    weight[keep_index] = perturbed_weight
    responsibility[keep_index] = perturbed_responsibility

    # Calculate log-likelihood.
    _, ll, dl = _expectation(y, mu, cov, weight, N_covpars)

    iterations = 1
    ll_dl = [(ll, dl)]

    while True:

        # Perform the maximization step.
        mu, cov, weight = _maximization(
            y, mu, cov, weight,
            responsibility, covariance_type, regularization,
            N_covpars)

        # Run the expectation step.
        responsibility, ll, dl = _expectation(
            y, mu, cov, weight, N_covpars)

        # Check for convergence.
        prev_ll, prev_dl = ll_dl[-1]
        relative_delta_ll = np.abs((ll - prev_ll)/prev_ll)

        ll_dl.append([ll, dl])
        iterations += 1

        if relative_delta_ll <= threshold \
        or iterations >= max_iterations:
            break


    meta = dict(warnflag=iterations >=max_iterations)
    if meta["warnflag"]:
        logger.warn("Maximum number of E-M iterations reached ({}) "\
                    "when merging components {} and {}".format(
                        max_iterations, a_index, b_index))

    meta["log-likelihood"] = ll

    return (mu, cov, weight, responsibility, meta, dl)


class GaussianMixtureEstimator(estimator.Estimator):

    r"""
    An estimator to model data from (potentially) multiple Gaussian
    distributions, using minimum message length.

    The priors and MML formalism is that of Figueiredo & Jain (2002).
    The score function and perturbation search strategy is that of Kasarapu
    & Allison (2015).

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param k_max: [optional]
        The initial (maximum) number of mixture components.
        If `None` is given, then :math:`N` will be assumed.

    :param k_min: [optional]
        The minimum number of mixture components.
        If `None` is given, the minimum number of mixtures is one.

    :param regularization: [optional]
        A regularizing factor to apply to covariance matrices. In very small
        samples, a regularization factor may be needed to apply to the
        diagonal of the covariance matrices (default: `0`).

    :param threshold: [optional]
        The relative improvement in log likelihood required before stopping
        the optimization process (default: `1e-5`)

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: `free`).
    """

    parameter_names = ("mean", "cov", "weight")

    def __init__(self, y, regularization=0,
        threshold=1e-5, covariance_type="free", **kwargs):

        super(GaussianMixtureEstimator, self).__init__(**kwargs)

        y = np.atleast_2d(y)
        
        if regularization < 0:
            raise ValueError("regularization strength must be non-negative")

        if threshold <= 0:
            raise ValueError("stopping threshold must be a positive value")

        available = ("free", "diag", "tied", "tied_diag")
        covariance_type = covariance_type.strip().lower()
        if covariance_type not in available:
            raise ValueError("covariance type '{}' is invalid. "\
                             "Must be one of: {}".format(
                                covariance_type, ", ".join(available)))

        if covariance_type not in ("free", "tied"):
            raise NotImplementedError("don't get your hopes up")

        self._y = y
        self._regularization = regularization
        self._threshold = threshold
        self._covariance_type = covariance_type

        return None


    @property
    def y(self):
        r""" Return the data values, :math:`y`. """
        return self._y


    @property
    def covariance_type(self):
        r""" Return the type of covariance stucture assumed. """
        return self._covariance_type


    @property
    def regularization(self):
        r""" Return the regularization strength. """
        return self._regularization


    def optimize(self):
        r"""
        Minimize the message length of a mixture of Gaussians, using the
        score function and perturbation search algorithm described by
        Kasarapu & Allison (2015).

        # TODO: more docs and returns
        """

        y = self._y # for convenience.
        N, D = y.shape
        N_cp = _component_covariance_parameters(D, self.covariance_type)

        # Initialize as a one-component mixture.
        mu, cov, weight = _initialize(y)

        iterations = 1
        responsibility, ll, mindl = _expectation(y, mu, cov, weight, N_cp)

        ll_dl = [(ll, mindl)]
        op_params = [mu, cov, weight]

        while True:

            M = weight.size
            best_perturbations = defaultdict(lambda: [np.inf])

            print("OUTER", iterations, M, mindl)

            # Exhaustively split all components.
            for m in range(M):
                r = (p_mu, p_cov, p_weight, p_responsibility, p_meta, p_dl) \
                  = _split_component(y, mu, cov, weight, responsibility, m)

                print("split", iterations, m, p_dl)

                # Keep best split component.
                if p_dl < best_perturbations["split"][-1]:
                    best_perturbations["split"] = [m] + list(r)
            
            if M > 1:
                # Exhaustively delete all components.
                for m in range(M):
                    r = (p_mu, p_cov, p_weight, p_responsibility, p_meta, p_dl) \
                      = _delete_component(y, mu, cov, weight, responsibility, m)

                    # Keep best deleted component.
                    if p_dl < best_perturbations["delete"][-1]:
                        best_perturbations["delete"] = [m] + list(r)
                
                # Exhaustively merge all components.
                for m in range(M):
                    r = (p_mu, p_cov, p_weight, p_responsibility, p_meta, p_dl) \
                      = _merge_component(y, mu, cov, weight, responsibility, m)

                    # Keep best merged component.
                    if p_dl < best_perturbations["merge"][-1]:
                        best_perturbations["merge"] = [m] + list(r)

            # Get best perturbation.
            operation, bp \
                = min(best_perturbations.items(), key=lambda x: x[1][-1])
            b_m, b_mu, b_cov, b_weight, b_responsibility, b_meta, b_dl = bp


            if mindl > b_dl:
                # Set the new state as the best perturbation.
                iterations += 1
                mindl = b_dl
                mu, cov, weight, responsibility \
                    = (b_mu, b_cov, b_weight, b_responsibility)

            else:
                # None of the perturbations were better than what we had.
                break

        op_params = (mu, cov, weight)
        return op_params, mindl
