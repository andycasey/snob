
""" An estimator for a mixture of Gaussians, using minimum message length. """

__all__ = ["GaussianMixtureEstimator"]

import logging
import numpy as np
import scipy.optimize as op

logger = logging.getLogger(__name__)

from . import estimator


def _number_of_covariance_parameters(D, covariance_type):
    """
    Return the number of parameters, given the number of observed dimensions
    and the covariance type.

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


def _evaluate_responsibility(y, mu, cov):
    """
    Return the distance between the data and a multivariate normal distribution.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The mean values of the multivariate normal distribution.

    :param cov:
        The covariance matrix of the multivariate normal distribution.
    """

    N, D = y.shape
    Cinv = np.linalg.inv(cov)
    scale = (2 * np.pi)**(-D/2.0) * np.linalg.det(cov)**(-0.5)
    d = y - mu
    return scale * np.exp(-0.5 * np.sum(d.T * np.dot(Cinv, d.T), axis=0))


def _evaluate_responsibilities(y, mu, cov):
    """
    Evaluate the responsibilities of :math:`K` components to the data.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The mean values of the :math:`K` multivariate normal distributions.

    :param cov:
        The covariance matrices of the :math:`K` multivariate normal
        distributions.

    :returns:
        The responsibility matrix.
    """

    K, N = (mu.shape[0], y.shape[0])
    responsibility = np.zeros((K, N))
    for k, (mu_k, cov_k) in enumerate(zip(mu, cov)):
        responsibility[k] = _evaluate_responsibility(y, mu_k, cov_k)
    return responsibility


def _kill_component(y, mu, cov, weight, component_index):
    """
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


def _initialize(y, K, scalar=0.10):
    """
    Return initial estimates of the parameters of the :math:`K` Gaussian
    components.

    :param y:
        The data values, :math:`y`.

    :param K:
        The number of Gaussian mixtures.

    :param scalar: [optional]
        The scalar to apply (multiplicatively) to the mean of the
        variances along each dimension of the data at initialization time.

    :returns:
        A three-length tuple containing the :math:`K` initial (multivariate)
        means, the :math:`K` covariance matrices, and the relative weights of
        each component.
    """

    N, D = y.shape
    random_indices = np.random.choice(np.arange(N), size=K, replace=False)
    
    mu = y[random_indices]
    cov = np.eye(D) * scalar * np.max(np.diag(np.cov(y.T)))
    cov = np.tile(cov, (K, 1)).reshape((K, D, D))
    weight = (1.0/K) * np.ones((K, 1))
    
    return (mu, cov, weight)

 
def _expectation(y, mu, cov, weight, N_covpars):
    """
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

    N, D = y.shape
    K = weight.size

    responsibility = _evaluate_responsibilities(y, mu, cov)
    log_likelihood = np.sum(np.log(np.sum(responsibility * weight, axis=0)))
    delta_length = -log_likelihood \
        + (N_covpars/2.0 * np.sum(np.log(weight))) \
        + (N_covpars/2.0 + 0.5) * K * np.log(N)

    return (responsibility, log_likelihood, delta_length)


def _score_function(members, N, N_covpars):
    """
    Return the score function (unnormalized weight) for a single component,
    as defined by Figueriedo & Jain (2002).

    :param members:
        The estimated number of members (total responsibility) for a component.

    :param N:
        The number of observations.

    :param N_covpars:
        The number of parameters due to the covariance structure.
    """
    return np.max([(members - N_covpars/2.0)/N, 0])


def _maximization(y, mu, cov, weight, responsibility, component_index, 
    covariance_type="free", regularization=0, N_covpars=None):
    """
    Perform the maximization step of the expectation-maximization algorithm on
    a single component.

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

    :param component_index:
        The index of the component to maximize.

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

    K = weight.size
    N, D = y.shape
    N_covpars = N_covpars or _number_of_covariance_parameters(D, covariance_type)

    indices = responsibility * weight
    memberships = indices / np.kron(np.ones((K, 1)), np.sum(indices, axis=0))
    normalization_constant = 1.0/np.sum(memberships[component_index])

    aux = np.kron(memberships[component_index], np.ones((D, 1))).T * y
    mu_k = normalization_constant * np.sum(aux, axis=0)

    if covariance_type in ("free", "tied"):
        emu = mu_k.reshape(-1, 1)
        cov_k = normalization_constant * np.dot(aux.T, y) \
              - np.dot(emu, emu.T) \
              + regularization * np.eye(D)
    else:
        cov_k = normalization_constant * np.diag(np.sum(aux * y, axis=0)) \
              - np.diag(mu_k**2)

    # Apply score function.
    unnormalized_weight_k \
        = _score_function(np.sum(memberships[component_index]), N, N_covpars)

    # Update parameters.
    # TODO: Create copy of mu, etc?
    mu[component_index] = mu_k
    cov[component_index] = cov_k
    weight[component_index] = unnormalized_weight_k

    # Re-normalize the weights.
    weight = weight / np.sum(weight)

    # Update responsibility matrix for this component.
    responsibility[component_index] = _evaluate_responsibility(y, mu_k, cov_k)

    return (mu, cov, weight, responsibility)


class GaussianMixtureEstimator(estimator.Estimator):

    """
    An estimator to model data from (potentially) multiple Gaussian
    distributions, using minimum message length.
    
    The priors, formalism, and search strategy given here is an implementation
    of Figueiredo & Jain (2002).

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

    parameter_names = ("mean", "cov", "fractions")

    def __init__(self, y, k_max=None, k_min=None, regularization=0, 
        threshold=1e-5, covariance_type="free", **kwargs):

        super(GaussianMixtureEstimator, self).__init__(**kwargs)

        y = np.atleast_2d(y)
        N, D = y.shape
        k_max, k_min = (k_max or N, k_min or 1)

        if k_max > N:
            raise ValueError(
                "I don't believe that you want more components than data")

        if k_min > k_max:
            raise ValueError("k_min must be less than k_max")

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
        self._k_max = k_max
        self._k_min = k_min
        self._regularization = regularization
        self._threshold = threshold
        self._covariance_type = covariance_type

        return None


    @property
    def y(self):
        """ Return the data values, :math:`y`. """
        return self._y


    @property
    def covariance_type(self):
        """ Return the type of covariance stucture assumed. """
        return self._covariance_type


    @property
    def regularization(self):
        """ Return the regularization strength. """
        return self._regularization


    @property
    def k_max(self):
        """ Return the maximum number of allowed components, :math:`K_{max}`. """
        return self._k_max


    @property
    def k_min(self):
        """ Return the minimum number of allowed components, :math:`K_{min}`. """
        return self._k_min


    def optimize(self, scalar=0.1):
        """
        Minimize the message length (cost function) by a variant of
        expectation maximization, as described in Figueiredo and Jain (2002).

        :param scalar: [optional]
            The scalar to apply (multiplicatively) to the mean of the
            variances along each dimension of the data at initialization time.

        :returns:
            A two-length tuple containing the optimized mixture parameters
            `(mu, cov, fractions)` and the log-likelihood for that set of
            mixtures.
        """

        N, D = self.y.shape
        N_cp = _number_of_covariance_parameters(D, self.covariance_type)

        # Initialization.
        mu, cov, weight = _initialize(self.y, self.k_max, scalar)

        # Calculate initial log-likelihood and message length offset
        R, ll, mindl = _expectation(self.y, mu, cov, weight, N_cp)
        ll_dl = [(ll, mindl)]

        op_params = [mu, cov, weight]
        
        while True:
            keep_splitting_components = True
            while keep_splitting_components:

                k = 0
                # Can't use a for loop here because K can become smaller by
                # killing away components.
                while weight.size > k: 

                    # Run maximization step on a single component.

                    # TODO: This could be parallelised, and then collect
                    #       components that won't be killed.

                    #       We would just need to store the parameters later
                    #       and update the responsibilities separately.
                    mu, cov, weight, R = _maximization(
                        self.y, mu, cov, weight, R, k, 
                        self.covariance_type, self.regularization, N_cp)

                    
                    # If the current component gets killed, 
                    # we have some paperwork to do.
                    kill_this_component = (weight[k] == 0)
                    if kill_this_component:
                        mu, cov, weight, R \
                            = _kill_component(self.y, mu, cov, weight, k)
                        # Leave k because we just killed the k-th component.

                    else:
                        # Iterate to the next component.
                        k += 1
                
                # Run expectation step.
                R, ll, dl = _expectation(self.y, mu, cov, weight, N_cp)
                ll_dl.append([ll, dl])

                # Compute change in the log likelihood to see if we should stop
                relative_delta_ll = (ll_dl[-1][0] - ll_dl[-2][0])/ll_dl[-2][0]
                if np.abs(relative_delta_ll) <= self._threshold \
                or weight.size == self._k_min:
                    keep_splitting_components = False

            if ll_dl[-1][1] < mindl:
                op_params = [mu, cov, weight]
                mindl = ll_dl[-1][1]

            # Should we kill off the smallest component and try again?
            if weight.size > self._k_min:
                smallest_component_index = np.argmin(weight)
                mu, cov, weight, R = _kill_component(
                    self.y, mu, cov, weight, smallest_component_index)
                
                R, ll, dl = _expectation(self.y, mu, cov, weight, N_cp)
                ll_dl.append([ll, dl])

            else:
                break # because weight.size = K = K_{min}

        index = np.argmin(np.array(ll_dl)[:, 1])
        ll, dl = ll_dl[index]

        self.mean, self.cov, self.weight = op_params

        return (op_params, ll)
