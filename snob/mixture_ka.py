
"""
An estimator for a mixture of Gaussians, using minimum message length.

The MML formalism is that of (Figueriedo & Jain, 2002), but the search
strategy and score function adopted is that of Kasarapu & Allison (2015).
"""

__all__ = [
    "GaussianMixture", 
    "kullback_leibler_for_multivariate_normals",
    "responsibility_matrix",
    "split_component", "merge_component", "delete_component", 
] 
import logging
import numpy as np
import scipy
from collections import defaultdict

logger = logging.getLogger(__name__)


class GaussianMixture(object):

    r"""
    Model data from (potentially) many multivariate Gaussian distributions, 
    using minimum message length (MML) as the cost function.

    The priors and MML formalism is that of Figueiredo & Jain (2002).
    The score function and perturbation search strategy is that of Kasarapu
    & Allison (2015).

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).
    """

    parameter_names = ("mean", "cov", "weight")

    def __init__(self, y, covariance_type="free", **kwargs):

        y = np.atleast_2d(y)
        if 1 > y.size:
            raise ValueError("no data!")
        
        available = ("free", "diag", "tied", "tied_diag")
        covariance_type = covariance_type.strip().lower()
        if covariance_type not in available:
            raise ValueError("covariance type '{}' is invalid. "\
                             "Must be one of: {}".format(
                                covariance_type, ", ".join(available)))

        if covariance_type not in ("free", "tied"):
            raise NotImplementedError("don't get your hopes up")

        self._y = y
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



    def _fit_kasarapu_allison(self, threshold=1e-3, max_em_iterations=10000, 
        **kwargs):
        r"""
        Minimize the message length of a mixture of Gaussians, using the
        score function and perturbation search algorithm described by
        Kasarapu & Allison (2015).

        :param threshold: [optional]
            The relative improvement in log likelihood required before stopping
            an expectation-maximization step (default: ``1e-5``).

        :param max_em_iterations: [optional]
            The maximum number of iterations to run per expectation-maximization
            loop (default: ``10000``).

        :returns:
            A tuple containing the optimized parameters ``(mu, cov, weight)``.
        """

        if 0 >= threshold:
            raise ValueError("threshold must be a positive value")

        if 1 > max_em_iterations:
            raise ValueError("max_em_iterations must be a positive integer")

        y = self._y # for convenience.
        N, D = y.shape
        
        callback = kwargs.get("callback", None)

        kwds = dict(
            threshold=threshold, 
            max_em_iterations=max_em_iterations,
            covariance_type=self.covariance_type, 
            callback=callback)

        # Initialize as a one-component mixture.
        mu, cov, weight = _initialize(y, callback=callback)

        iterations = 1
        N_cp = _parameters_per_component(D, self.covariance_type)
        R, ll, mindl = _expectation(y, mu, cov, weight, N_cp)

        ll_dl = [(ll, mindl)]
        op_params = [mu, cov, weight]

        while True:

            M = weight.size
            best_perturbations = defaultdict(lambda: [np.inf])

            # Exhaustively split all components.
            for m in range(M):
                r = (_mu, _cov, _weight, _R, _meta, p_dl) \
                  = split_component(y, mu, cov, weight, R, m, **kwds)

                # Keep best split component.
                if p_dl < best_perturbations["split"][-1]:
                    best_perturbations["split"] = [m] + list(r)
            
            if M > 1:
                # Exhaustively delete all components.
                for m in range(M):
                    r = (_mu, _cov, _weight, _R, _meta, p_dl) \
                      = delete_component(y, mu, cov, weight, R, m, **kwds)

                    # Keep best deleted component.
                    if p_dl < best_perturbations["delete"][-1]:
                        best_perturbations["delete"] = [m] + list(r)
                
                # Exhaustively merge all components.
                for m in range(M):
                    r = (_mu, _cov, _weight, _R, _meta, p_dl) \
                      = merge_component(y, mu, cov, weight, R, m, **kwds)

                    # Keep best merged component.
                    if p_dl < best_perturbations["merge"][-1]:
                        best_perturbations["merge"] = [m] + list(r)

            # Get best perturbation.
            operation, bp \
                = min(best_perturbations.items(), key=lambda x: x[1][-1])
            b_m, b_mu, b_cov, b_weight, b_R, b_meta, b_dl = bp

            if mindl > b_dl:
                # Set the new state as the best perturbation.
                iterations += 1
                mindl = b_dl
                mu, cov, weight, R \
                    = (b_mu, b_cov, b_weight, b_R)

            else:
                # None of the perturbations were better than what we had.
                break

        # TODO: a full_output response.
        return (mu, cov, weight)
        
    fit = _fit_kasarapu_allison


def _old_responsibility_matrix(y, mu, cov, weight, full_output=False):
    r"""
    Return the responsibility matrix,

    .. math::

        r_{ij} = \frac{w_{j}f\left(x_i;\theta_j\right)}{\sum_{k=1}^{K}{w_k}f\left(x_i;\theta_k\right)}


    where :math:`r_{ij}` denotes the conditional probability of a datum
    :math:`x_i` belonging to the :math:`j`-th component. The effective 
    membership associated with each component is then given by

    .. math::

        n_j = \sum_{i=1}^{N}r_{ij}
        \textrm{and}
        \sum_{j=1}^{M}n_{j} = N


    where something.
    
    :param y:
        The data values, :math:`y`.

    :param mu:
        The mean values of the :math:`K` multivariate normal distributions.

    :param cov:
        The covariance matrices of the :math:`K` multivariate normal
        distributions.

    :param weight:
        The current estimates of the relative mixing weight.

    :param full_output: [optional]
        If ``True``, return the responsibility matrix, the unnormalized
        responsibility matrix -- the numerator in the equation above -- and 
        the normalization constants -- the denominator in the equation above
        (default: ``False``).

    :returns:
        The responsibility matrix. If ``full_output=True``, then a
        three-length tuple will be returned that contains: the responsibility
        matrix, the unnormalized responsibility matrix (the numerator in the 
        equation above), and the normalization constants (the denominator in 
        the equation above).
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

    numerator[~np.isfinite(numerator)] = 0.0

    eps = (7./3 - 4./3 - 1) # machine precision
    denominator = np.clip(np.sum(numerator, axis=0), eps, np.inf)
    responsibility = np.clip(numerator/denominator, eps, 1)

    assert np.all(np.isfinite(denominator))
    assert np.all(denominator < 10e8)
    assert np.all(np.isfinite(responsibility))

    responsibility /= np.sum(responsibility, axis=0)

    #foo = _new_responsibility_matrix(y, mu, cov, weight)

    return (responsibility, numerator, denominator) if full_output \
                                                    else responsibility


def responsibility_matrix(y, mu, cov, weight, full_output=False):
    r"""
    Return the responsibility matrix,

    .. math::

        r_{ij} = \frac{w_{j}f\left(x_i;\theta_j\right)}{\sum_{k=1}^{K}{w_k}f\left(x_i;\theta_k\right)}


    where :math:`r_{ij}` denotes the conditional probability of a datum
    :math:`x_i` belonging to the :math:`j`-th component. The effective 
    membership associated with each component is then given by

    .. math::

        n_j = \sum_{i=1}^{N}r_{ij}
        \textrm{and}
        \sum_{j=1}^{M}n_{j} = N


    where something.
    
    :param y:
        The data values, :math:`y`.

    :param mu:
        The mean values of the :math:`K` multivariate normal distributions.

    :param cov:
        The covariance matrices of the :math:`K` multivariate normal
        distributions.

    :param weight:
        The current estimates of the relative mixing weight.

    :param full_output: [optional]
        If ``True``, return the responsibility matrix, and the log likelihood,
        which is evaluated for free (default: ``False``).

    :returns:
        The responsibility matrix. If ``full_output=True``, then the
        log likelihood (per observation) will also be returned.
    """

    N, D = y.shape
    K = mu.shape[0]


    constant = -D / 2.0 * np.log(2 * np.pi)
    log_responsibility = constant * np.ones((K, N))

    for k, (mu_k, cov_k) in enumerate(zip(mu, cov)):

        U, S, V = np.linalg.svd(cov_k)
        Cinv = np.dot(np.dot(V.T, np.linalg.inv(np.diag(S))), U.T)

        O = y - mu_k
        log_responsibility[k] += np.log(weight[k]) - 0.5 * (
            np.log(np.linalg.det(cov_k))
          + np.sum(O.T * np.dot(Cinv, O.T), axis=0))

    # Any NaNs in log_responsibility occur because of almost infinitely small
    # probability.
    assert np.all(np.isfinite(log_responsibility))
    #log_responsibility[~np.isfinite(log_responsibility)] = -100
    
    log_likelihood = scipy.misc.logsumexp(log_responsibility, axis=0)

    responsibility = np.exp(log_responsibility - log_likelihood)

    assert np.all(np.isfinite(responsibility))

    return (responsibility, log_likelihood) if full_output else responsibility


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


def _parameters_per_component(D, covariance_type):
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

    :returns:
        The number of parameters required to fully specify the multivariate
        mean and covariance matrix of :math:`D` Gaussian distributions.
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


def _initialize(y, callback=None):
    r"""
    Return initial estimates of the parameters.

    :param y:
        The data values, :math:`y`.

    :returns:
        A three-length tuple containing the initial (multivariate) mean,
        the covariance matrix, and the relative weight.
    """

    weight = np.ones((1, 1))
    N, D = y.shape
    means = np.mean(y, axis=0).reshape((1, -1))
    cov = np.cov(y.T).reshape((1, D, D))

    state = (means, cov, weight)
    if callback is not None:
        callback(state, dict(stage="initialization"))
    
    return state


def _expectation(y, mu, cov, weight, N_component_pars, **kwargs):
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

    :param N_component_pars:
        The number of parameters required to specify the mean and covariance
        matrix of a single Gaussian component.

    :returns:
        A three-length tuple containing the responsibility matrix,
        the log likelihood, and the change in message length.
    """

    """
    from time import time
    
    if weight.size > 2:
        ta = time()
        foo = responsibility_matrix(y, mu, cov, weight, full_output=True)
        tb = time()
        bar = _new_responsibility_matrix(y, mu, cov, weight, full_output=True)
        tc = time()
        print(tb - ta)
        print(tc - tb)
        raise a
    """

    responsibility, log_likelihood = responsibility_matrix(
        y, mu, cov, weight, full_output=True)

    
    # Eq. 40 omitting -Nd\log\eps
    #log_likelihood = np.sum(log_likelihood) 

    # TODO: check delta_length.
    # N_component_pars = int(D + D*(D + 1)/2.0)
    N, D = y.shape
    K = weight.size
    delta_length = -np.sum(log_likelihood) \
        + (N_component_pars/2.0 * np.sum(np.log(weight))) \
        + (N_component_pars/2.0 + 0.5) * K * np.log(N)

    # I(K) = K\log{2} + constant

    # Eq. 38
    # I(w) = (M-1)/2 * log(N) - 0.5\sum_{k=1}^{K}\log{w_k} - (K - 1)!

    delta_length2 = _message_length(D, cov, weight, responsibility, log_likelihood)

    print("DELTA_LENGTHS", delta_length, delta_length2)


    return (responsibility, -np.sum(log_likelihood), delta_length2)

def old_log_kappa(D):
    r"""
    Return an approximation of the logarithm of the lattice constant 
    :math:`\kappa_D` using the relationship:

    .. math::

        I_1(x) \approx - \log\frac{h(\theta')}{\sqrt{F(\theta')}} 
                       - \log{f(x|\theta')}
                       - (D/2)\log{2\pi}
                       + \frac{1}{2}\log{(\pi{}D)}
                       - 1

    The expected error on :math:`\kappa_D` is less than 0.1 nit.  See Sections
    5.2.12 and 3.3.4 of Wallace (2005) for more details.

    :param D:
        The number of dimensions.
    """

    return  -0.5*D*np.log(2*np.pi) + 0.5*np.log(D*np.pi) - 1


def log_kappa(D):

    cd = -0.5 * D * np.log(2 * np.pi) + 0.5 * np.log(D * np.pi)
    return -1 + 2 * cd/D

def _message_length(D, cov, weight, responsibility, log_likelihood=None, eps=0.10):

    M, N = responsibility.shape
    
    # I(M) = M\log{2} + constant
    I_m = M # [bits]

    # I(w) = \frac{(M - 1)}{2}\log{N} - \frac{1}{2}\sum_{j=1}^{M}\log{w_j} - (M - 1)!
    I_w = (M - 1) / 2.0 * np.log(N) \
        - 0.5 * np.sum(np.log(weight)) \
        - np.math.factorial(M - 1) \
        + 1
    I_w = I_w/np.log(2) # [bits]


    # \sum_{j=1}^{M}I(\theta_j) = -\sum_{j=1}^{M}\log{h\left(\theta_j\right)}
    #                           + 0.5\sum_{j=1}^{M}\log\det{F\left(\theta_j\right)}

    # \det{F\left(\mu,C\right)} \approx \det{F(\mu)}\dot\det{F(C)}
    # |F(\mu)| = N^{d}|C|^{-1}
    # |F(C)| = N^\frac{d(d + 1)}{2}2^{-d}|C|^{-(d+1)}

    # thus |F(\mu,C)| = N^\frac{d(d+3)}{2}\dot{}2^{-d}\dot{}|C|^{-(d+2)}

    # old:
    """
    log_F_m = 0.5 * D * (D + 3) * np.log(N) \
            - 2 * np.log(D) \
            - (D + 2) * np.nansum(np.log(np.linalg.det(cov)))
    """


    # For multivariate, from Kasaraup:
    """
    log_F_m = 0.5 * D * (D + 3) * np.log(N) # should be Neff? TODO
    log_F_m += -np.sum(log_det_cov)
    log_F_m += -(D * np.log(2) + (D + 1) * np.sum(log_det_cov))    
    """
    if D == 1:
        log_F_m = np.log(2) + (2 * np.log(N)) - 4 * np.log(cov.flatten()[0]**0.5)
    else:
        log_det_cov = np.log(np.linalg.det(cov))
    
        log_F_m = 0.5 * D * (D + 3) * np.log(N) 
        log_F_m += -np.sum(log_det_cov)
        log_F_m += -(D * np.log(2) + (D + 1) * np.sum(log_det_cov))    
        
        
    # TODO: No prior on h(theta).. thus -\sum_{j=1}^{M}\log{h\left(\theta_j\right)} = 0

    # TODO: bother about including this? -N * D * np.log(eps)
    assert log_likelihood is not None





    N_cp = _parameters_per_component(D, "free")
    part2 = -np.sum(log_likelihood) + N_cp/(2*np.log(2))

    AOM = 0.001 # MAGIC
    Il = -np.sum(log_likelihood) - (D * N * np.log(AOM))
    Il = Il/np.log(2) # [bits]

    if D == 1:

        # R1
        R1 = 10 # MAGIC
        R2 = 2 # MAGIC
        log_prior = D * np.log(R1) # mu
        log_prior += np.log(R2)
        log_prior += np.log(cov.flatten()[0]**0.5)

    
    else:
        R1 = 10
        log_prior = D * np.log(R1) + 0.5 * (D + 1) * np.sum(np.log(np.linalg.det(cov)))


    I_t = (log_prior + 0.5 * log_F_m)/np.log(2)


    num_free_params = (0.5 * D * (D+3) * M) + (M - 1)
    lattice = 0.5 * num_free_params * log_kappa(num_free_params) / np.log(2)


    part1 = I_m + I_w + I_t + lattice
    part2 = Il + (0.5 * num_free_params)/np.log(2)

    I = part1 + part2

    
    return I
    if I > 125:
        raise a

    I = I_m + I_w + (0 + 0.5 * log_F_m) - np.sum(log_likelihood) + lattice
    #assert I > 0
    assert np.isfinite(I)
    if weight.size > 1:
        raise a
    return I


def _maximization(y, mu, cov, weight, responsibility, parent_responsibility=1,
    **kwargs):
    r"""
    Perform the maximization step of the expectation-maximization algorithm
    on all components.

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
    
    :param parent_responsibility: [optional]
        An array of length :math:`N` giving the parent component 
        responsibilities (default: ``1``). Only useful if the maximization
        step is to be performed on sub-mixtures with parent responsibilities.

    :returns:
        A three length tuple containing the updated multivariate mean values,
        the updated covariance matrices, and the updated mixture weights. 
    """

    M = weight.size 
    N, D = y.shape
    
    # Update the weights.
    effective_membership = np.sum(responsibility, axis=1)
    new_weight = (effective_membership + 0.5)/(N + M/2.0)

    w_responsibility = parent_responsibility * responsibility
    w_effective_membership = np.sum(w_responsibility, axis=1) + 1e-16

    new_mu = np.zeros_like(mu)
    new_cov = np.zeros_like(cov)
    for m in range(M):
        new_mu[m] = np.sum(w_responsibility[m] * y.T, axis=1) \
                  / w_effective_membership[m]

        offset = y - new_mu[m]
        new_cov[m] = np.dot(w_responsibility[m] * offset.T, offset) \
                   / (w_effective_membership[m] - 1) \
                   + 1e-16 * np.eye(D)

    state = (new_mu, new_cov, new_weight)

    assert np.all(np.isfinite(new_mu))
    assert np.all(np.isfinite(new_cov))
    assert np.all(np.isfinite(new_weight))


    callback = kwargs.get("callback", None)
    if callback is not None:
        try:
            callback(state, "maximization")
        
        except:
            logger.exception("Exception when running callback:")
            raise

    return state 


def _expectation_maximization(y, mu, cov, weight, responsibility=None,
    covariance_type="free", threshold=1e-5, max_em_iterations=10000, **kwargs):
    r"""
    Run the expectation-maximization algorithm on the current set of
    multivariate Gaussian mixtures.

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

    :param responsibility: [optional]
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component. If ``None`` is given
        then the responsibility matrix will be calculated in the first
        expectation step.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :param threshold: [optional]
        The relative improvement in log likelihood required before stopping
        an expectation-maximization step (default: ``1e-5``).

    :param max_em_iterations: [optional]
        The maximum number of iterations to run per expectation-maximization
        loop (default: ``10000``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """   

    M = weight.size
    N, D = y.shape
    N_component_pars = _parameters_per_component(D, covariance_type)
    
    # Calculate log-likelihood and initial expectation step.
    _init_responsibility, ll, dl = _expectation(y, mu, cov, weight, N_component_pars)
    if responsibility is None:
        responsibility = _init_responsibility

    iterations = 1
    ll_dl = [(ll, dl)]

    while True:

        # Perform the maximization step.
        mu, cov, weight \
            = _maximization(y, mu, cov, weight, responsibility, **kwargs)

        # Run the expectation step.
        responsibility, ll, dl \
            = _expectation(y, mu, cov, weight, N_component_pars, **kwargs)

        # Check for convergence.
        prev_ll, prev_dl = ll_dl[-1]
        relative_delta_message_length = np.abs((ll - prev_ll)/prev_ll)
        ll_dl.append([ll, dl])
        iterations += 1

        assert np.isfinite(relative_delta_message_length)

        if relative_delta_message_length <= threshold \
        or iterations >= max_em_iterations:
            break

    meta = dict(warnflag=iterations >= max_em_iterations, log_likelihood=ll)
    if meta["warnflag"]:
        logger.warn("Maximum number of E-M iterations reached ({}) {}".format(
            max_em_iterations, kwargs.get("_warn_context", "")))

    return (mu, cov, weight, responsibility, meta, dl)


def split_component(y, mu, cov, weight, responsibility, index, 
    covariance_type="free", **kwargs):
    r"""
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

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be split.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """

    M = weight.size
    N, D = y.shape
    
    # Compute the direction of maximum variance of the parent component, and
    # locate two points which are one standard deviation away on either side.
    U, S, V = np.linalg.svd(cov[index])
    sigma = np.diag(cov[index])**0.5
    #if not np.all(np.isfinite(sigma)):
    #    # TODO: FAIL
    #    sigma = 1.0
    assert np.all(np.isfinite(sigma))

    child_mu = mu[index] + np.vstack([+V[0], -V[0]]) * sigma
    #np.diag(cov[index])**0.5
    assert np.all(np.isfinite(child_mu))

    raise a
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
    #eps = (7./3 - 4./3 - 1) # machine precision
    parent_responsibility = responsibility[index]

    # Run expectation-maximization on the child mixtures.
    child_mu, child_cov, child_weight, child_responsibility, meta, dl = \
        _expectation_maximization(y, child_mu, child_cov, child_weight, 
            #responsibility=child_responsibility, 
            parent_responsibility=parent_responsibility,
            covariance_type=covariance_type, **kwargs)

    
    # After the chld mixture is locally optimized, we need to integrate it
    # with the untouched M - 1 components to result in a M + 1 component
    # mixture M'.

    # An E-M is finally carried out on the combined M + 1 components to
    # estimate the parameters of M' and result in an optimized 
    # (M + 1)-component mixture.

    # Update the component weights.
    # Note that the child A mixture will remain in index `index`, and the
    # child B mixture will be appended to the end.

    raise a

    if M > 1:
        # Integrate the M + 1 components and run expectation-maximization
        weight = np.hstack([weight, [parent_weight * child_weight[1]]])
        weight[index] = parent_weight * child_weight[0]

        responsibility = np.vstack([responsibility, 
            [parent_responsibility * child_responsibility[1]]])
        responsibility[index] = parent_responsibility * child_responsibility[0]
        
        mu = np.vstack([mu, [child_mu[1]]])
        mu[index] = child_mu[0]

        cov = np.vstack([cov, [child_cov[1]]])
        cov[index] = child_cov[0]

        return _expectation_maximization(y, mu, cov, weight, responsibility, 
            covariance_type=covariance_type, **kwargs)
    
    else:
        # Simple case where we don't have to re-run E-M because there was only
        # one component to split.
        return (child_mu, child_cov, child_weight, child_responsibility, meta, dl)
    

def delete_component(y, mu, cov, weight, responsibility, index, 
    covariance_type="free", **kwargs):
    r"""
    Delete a component from the mixture, and return the new optimal state.

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

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be deleted.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """

    # Create new component weights.
    parent_weight = weight[index]
    parent_responsibility = responsibility[index]
    
    # Eq. 54-55
    new_weight = np.clip(
        np.delete(weight, index, axis=0)/(1-parent_weight),
        0, 1)
    
    # Calculate the new responsibility safely.
    new_responsibility = np.clip(
        np.delete(responsibility, index, axis=0) / (1 - parent_responsibility),
        0, 1)
    new_responsibility[~np.isfinite(new_responsibility)] = 0.0

    assert np.all(np.isfinite(new_responsibility))
    assert np.all(np.isfinite(new_weight))

    new_mu = np.delete(mu, index, axis=0)
    new_cov = np.delete(cov, index, axis=0)

    # Run expectation-maximizaton on the perturbed mixtures. 
    return _expectation_maximization(y, new_mu, new_cov, new_weight, 
        new_responsibility, covariance_type=covariance_type, **kwargs)


def merge_component(y, mu, cov, weight, responsibility, index, 
    covariance_type="free", **kwargs):
    r"""
    Merge a component from the mixture with its "closest" component, as
    judged by the Kullback-Leibler distance.

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

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be deleted.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """

    # Calculate the Kullback-Leibler distance to the other distributions.
    D_kl = np.inf * np.ones(weight.size)
    for m in range(weight.size):
        if m == index: continue
        D_kl[m] = kullback_leibler_for_multivariate_normals(
            mu[index], cov[index], mu[m], cov[m])

    a_index, b_index = (index, np.nanargmin(D_kl))

    # Initialize.
    weight_k = np.sum(weight[[a_index, b_index]])
    responsibility_k = np.sum(responsibility[[a_index, b_index]], axis=0)
    effective_membership_k = np.sum(responsibility_k)

    mu_k = np.sum(responsibility_k * y.T, axis=1) / effective_membership_k

    offset = y - mu_k
    cov_k = np.dot(responsibility_k * offset.T, offset) \
          / (effective_membership_k - 1)

    # Delete the b-th component.
    del_index = np.max([a_index, b_index])
    keep_index = np.min([a_index, b_index])

    new_mu = np.delete(mu, del_index, axis=0)
    new_cov = np.delete(cov, del_index, axis=0)
    new_weight = np.delete(weight, del_index, axis=0)
    new_responsibility = np.delete(responsibility, del_index, axis=0)

    new_mu[keep_index] = mu_k
    new_cov[keep_index] = cov_k
    new_weight[keep_index] = weight_k
    new_responsibility[keep_index] = responsibility_k

    # Calculate log-likelihood.
    return _expectation_maximization(y, new_mu, new_cov, new_weight,
        responsibility=new_responsibility, covariance_type=covariance_type,
        **kwargs)