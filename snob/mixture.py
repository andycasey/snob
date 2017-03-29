
""" An estimator for a mixture of Gaussians, using minimum message length. """

__all__ = ["GaussianMixtureEstimator"]

import logging
import numpy as np
import scipy.optimize as op

logger = logging.getLogger(__name__)

from . import estimator


def _number_of_params(D, covariance_type):
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


def evaluate_multinorm(y, mu, cov):
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
    r = y - mu
    return scale * np.exp(-0.5 * np.sum(r.T * np.dot(Cinv, r.T), axis=0))


def _kill_component(y, mu, cov, fractions, component):
    """
    Kill off a component, and return the new estimates of the component
    parameters.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param fractions:
        The current estimates of the relative mixing fractions.

    :param component:
        The index of the component to be killed off.
    """

    mu = np.delete(mu, component, axis=0)
    cov = np.delete(cov, component, axis=0)
    fractions = np.delete(fractions, component, axis=0)

    K, N = (mu.shape[0], y.shape[0])

    # Normalize the fractions.
    fractions = fractions/np.sum(fractions)

    semi_indices = np.zeros((K, N))
    for k in range(K):
        semi_indices[k] = evaluate_multinorm(y, mu[k], cov[k])
    
    return (mu, cov, fractions, semi_indices)
        
 
def _log_likelihood(semi_indices, fractions, N_pars):
    """
    Return the log-likelihood of the data, and the delta change in the message
    length.

    :param semi_indices:
        An array of semi_indices.

    :param fractions:
        The relative mixing fractions for each component.

    :param N_pars:
        The number of parameters due to the covariance structure.
    """

    K, N = semi_indices.shape

    log_likelihood = np.sum(np.log(np.sum(semi_indices * fractions, axis=0)))
    delta_length = -log_likelihood \
        + (N_pars/2.0 * np.sum(np.log(fractions))) \
        + (N_pars/2.0 + 0.5) * K * np.log(N)

    return (log_likelihood, delta_length)


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
    """

    N, D = y.shape
    random_indices = np.random.choice(np.arange(N), size=K, replace=False)
    
    mu = y[random_indices]
    fractions = (1.0/K) * np.ones((K, 1))

    global_cov = np.cov(y.T)
    cov = np.zeros((K, D, D))
    semi_indices = np.zeros((K, N))

    for k in range(K):
        cov[k] = np.eye(D) * scalar * np.max(np.diag(global_cov))
        semi_indices[k] =  evaluate_multinorm(y, mu[k], cov[k])
        
    return (mu, cov, fractions, semi_indices)


def _m_step(y, semi_indices, est_pp, component, D, covariance_type, 
    regularization, N_pars):
    """
    Perform the M-step (Section 5.1 of Figueiredo and Jain (2002) on a
    component and return estimates of the component parameters.
    """

    K, N = semi_indices.shape

    indices = semi_indices * est_pp
    norm_indices = \
          indices \
        / np.kron(np.ones((K, 1)), np.sum(indices, axis=0))

    normalize = 1.0/np.sum(norm_indices[component])
    aux = np.kron(norm_indices[component], np.ones((D, 1))).T * y
    mu = normalize * np.sum(aux, axis=0)

    if covariance_type in ("free", "tied"):
        emu = mu.reshape(-1, 1)
        cov = normalize * np.dot(aux.T, y) \
            - np.dot(emu, emu.T) \
            + regularization * np.eye(D)
    else:
        cov = normalize * np.diag(np.sum(aux * y, axis=0)) \
            - np.diag(mu**2)

    # Score function.
    pp = (1.0/N) * np.max([
        np.sum(norm_indices[component]) - N_pars/2.0,
        0
    ])

    return (mu, cov, pp)
    

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

        K = self._k_max
        N, D = self.y.shape
        N_pars = _number_of_params(D, self._covariance_type)

        # Initialization.
        est_mu, est_cov, est_pp, semi_indices = _initialize(self.y, K, scalar)

        # Calculate initial log-likelihood and message length offset
        ll, mindl = _log_likelihood(semi_indices, est_pp, N_pars)
        ll_dl = [(ll, mindl)]

        best = []
        
        while True:
            keep_splitting_components = True
            while keep_splitting_components:

                component = 0
                # Can't use a for loop here because K can be made smaller.
                while K > component: 

                    # TODO: this could be parallelised, and then collect
                    #       components that won't be killed
                    component_mu, component_cov, component_pp = _m_step(
                        self.y, semi_indices, est_pp, component, D, 
                        self._covariance_type, self._regularization, N_pars)

                    kill_this_component = (component_pp == 0)

                    # If the current component gets killed, we have some
                    # paperwork to do.
                    if kill_this_component:
                        K -= 1
                        est_mu, est_cov, est_pp, semi_indices \
                            = _kill_component(
                                self.y, est_mu, est_cov, est_pp, component)
                        
                    else:
                        # Store the new mu, cov, and pp for this component.
                        est_mu[component], est_cov[component], est_pp[component] \
                            = (component_mu, component_cov, component_pp)
                        
                        # Re-normalize fractions.
                        est_pp = est_pp/np.sum(est_pp)

                        # Update the corresponding indicator variable.
                        semi_indices[component] = evaluate_multinorm(
                            self.y, component_mu, component_cov)
                        
                        # Iterate to the next component.
                        component += 1

                # TODO: parallelisable
                for k in range(K):
                    semi_indices[k] \
                        = evaluate_multinorm(self.y, est_mu[k], est_cov[k])
                
                ll_dl.append(_log_likelihood(semi_indices, est_pp, N_pars))
                
                # Compute change in the log likelihood to see if we should stop
                relative_delta_ll = (ll_dl[-1][0] - ll_dl[-2][0])/ll_dl[-2][0]
                if np.abs(relative_delta_ll) <= self._threshold:
                    keep_splitting_components = False

            if ll_dl[-1][1] < mindl:
                best = [est_mu, est_cov, est_pp]            
                mindl = ll_dl[-1][1]

            # Should we kill off a component and try again?
            if est_mu.shape[0] > self._k_min:

                K -= 1
                est_mu, est_cov, est_pp, semi_indices = _kill_component(
                    self.y, est_mu, est_cov, est_pp, np.argmin(est_pp))

                ll_dl.append(_log_likelihood(semi_indices, est_pp, N_pars))
                
            else:
                break # because K = K_{min}

        index = np.argmin(np.array(ll_dl)[:, 1])
        ll, dl = ll_dl[index]

        self.mean, self.cov, self.fractions = best

        return (best, ll)
