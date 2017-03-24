
""" An estimator for a single multivariate latent factor. """

__all__ = ["SingleLatentFactorEstimator"]

import logging
import numpy as np
import scipy.optimize as op

from . import estimator

logger = logging.getLogger(__name__)


def _log_prior(sigma, a):
    r"""
    Return the natural logarithm of the prior.

    The priors on the means :math:`\mu_k` are assumed to be uniform over
    some bounded range, if given, and that region is expected to be large
    compared to :math:`a_k`.

    The prior on :math:`\sigma_k` has a density proportional to 
    :math:`1/\sigma_k` over some (given) finite range.

    For the prior on :math:`\underline{b}` (where :math:`b_k = a_k/\sigma_k`)
    we assume all directions in :math:`K`-space to be equally likely, and 
    a prior density in :math:`K`-space proportional to:

    .. math::

        p(b) = (1 + b^2)^{-(K + 1)/2}

    which expresses the expectation that in each dimension, :math:`a_k`
    will be of the same order as :math:`\sigma_k` but could be considerably
    larger. For :math:`b^2 < 1`, it is slowly varying. For :math:`b^2 \gg 1`,
    it leads to a density for :math:`b = |\underline{b}|` proportional to
    :math:`1/b^2`. The resulting prior :math:`p(a|\sigma)` is:

    .. math::

        p(a|\sigma) = (1 + b^2)^{-(K + 1)/2} \prod_{k}\frac{1}{\sigma_k}
    
    Excluding the bounds on :math:`\mu_k`, 
    the total logarithm of the prior is:
    # TODO

    .. math::

        \log{p(\theta)} = -2\sum_{k}\log{\sigma_k}
                          -\frac{1}{2}(K + 1)\log{1 + b^2}
    """
    K = sigma.size
    sls = np.sum(np.log(sigma))
    b_squared = np.sum((a/sigma)**2)

    return np.sum([
        -sls, # \log{p(\sigma)}
        -0.5 * (K + 1) * np.log(1 + b_squared) - sls, # \log{p(a|\sigma)}
    ])


def _log_likelihood(y, mean, sigma, v, a):
    r"""
    Return the natural log likelihood of the data.

    From Section 6.9 of Wallace (2005), the *negative* log likelihood is:

    .. math::

        -log{f(y|\theta)} = \frac{1}{2}KN\log(2\pi)
                          + N\sum_{k}\log{\sigma_k}
                          + \frac{1}{2}\sum_{n}\sum_{k}(y_nk} - \mu_k - v_{n}a_{k})^2/\sigma_{k}^2
    
    :param y:
        A two-dimensional array with shape :math:`(N, K)` consisting of
        :math:`K` measured attributes from :math:`N` independent observations.

    :param mean:
        An :math:`K`-length array of mean values :math:`\mu_k`.

    :param sigma:
        An :math:`K`-length array of Gaussian uncertainties :math:`\sigma_k`.

    :param v:
        An :math:`N`-length array containing the factor scores :math:`v_n`.

    :param a:
        An :math:`K`-length array containing the factor loads :math:`a_k`.
    """

    N, K = y.shape
    v_dot_a = np.dot(v.reshape(N, 1), a.reshape(1, K))
    return -np.sum([
        0.5*N*K*np.log(2*np.pi),
        N * np.sum(np.log(sigma)),
        0.5 * np.sum((y - mean - v_dot_a)**2 / sigma**2)
    ])


def _log_fisher(sigma, v, a):
    r"""
    Return the natural logarithm of the Fisher information.

    If :math:``K \ll N``, then Wallace (2005, p. 298) shows that the Fisher 
    information is:

    .. math::

        F(\underline{\mu},\underline{\sigma},\underline{a},\underline{v})
            = (2N)^{K}(Nv^2 - S^2)^{K}(1 + b^2)^{N-2} / \prod_{k}\sigma_{k}^6

    where :math:`S = \sum_{n}v_n` and :math:`b^2 = \underline{b}^2 = \sum_{k}b_k^2`
    and :math:`b_k = a_k/\sigma_k`.

    :param sigma:
        An :math:`K`-length array of Gaussian uncertainties :math:`\sigma_k`.

    :param v:
        An :math:`N`-length array containing the factor scores :math:`v_n`.

    :param a:
        An :math:`K`-length array containing the factor loads :math:`a_k`.
    """

    N, K = (len(v), len(sigma))
    v_squared = np.sum(v**2)
    S_squared = np.sum(v)**2
    b_squared = np.sum((a/sigma)**2)
    
    return np.prod([ # for code legibility
        (2*N)**K,
        (N * v_squared - S_squared)**K,
        (1 + b_squared)**(N - 2),
        np.prod(sigma**-6)
    ])


def _unpack_parameters(parameters, N, K):

    mean = parameters[:K]
    sigma = parameters[K:2*K]
    v = parameters[2*K:2*K + N] # factor_scores, v
    a = parameters[2*K + N:] # factor_loads, a

    return (mean, sigma, v, a)


def _message_length(parameters, y):
    """
    Return the approximate message length (omitting constant terms) of a 
    single (multivariate) latent factor estimator, given the parameters.

    # TODO
    """

    N, K = y.shape
    assert parameters.size == (3 * K + N)

    mean, sigma, v, a = _unpack_parameters(parameters, N, K)

    v_squared = np.sum(v**2)
    S_squared = np.sum(v)**2
    b_squared = np.sum((a/sigma)**2)

    v_dot_a = np.dot(v.reshape(N, 1), a.reshape(1, K))

    foo = np.sum([
        (N - 1) * np.sum(np.log(sigma)),
        0.5 * (K*np.log(N*v_squared-S_squared) + (N+K-1)*np.log(1 + b_squared)),
        0.5 * (v_squared + np.sum((y - mean - v_dot_a)**2 / sigma**2))
    ])
    # TODO
    print(parameters, foo)
    return foo


class SingleLatentFactorEstimator(estimator.Estimator):

    r"""
    An estimator to model data with a single multivariate latent factor,
    or rather, as a multivariate normal distribution with a special covariance
    structure.

    The data are :math:`N` independent observations from a :math:`K`
    dimensional distribution.

    .. math::

        y &= \{\underline{y}_n: n=1,\dots,N\}\\
        \underline{y}_n &= \left(y_{nk}: k = 1,\dots,K\right)

    The assumed model is:

    .. math::

        y_{nk} = \mu_k + v_{n}a_{k} + \sigma_{k}r_{nk}


    where the variates :math:`\{r_{nk}: k=1,\dots,K; n=1,\dots,N\}` are
    all independent and identically distributed variates from :math:`N(0, 1)`.

    This estimator uses quadratic approximations to strict minimum message
    length.

    .. note::

        A latent factor model is described in Section 6.9 of Wallace (2005).
        However Wallace (2005) uses :math:`x` to describe the data, and
        defines :math:`y_{nk} = w_{nk}/\sigma_k`. In order for this code to
        be more self-consistent in its nomenclature, we have adopted :math:`y`
        to describe the data, and the ratios :math:`w_{nk}/\sigma_k` are
        internal attributes that do not need to be exposed to the user.
    
    :param y:
        A two-dimensional array with shape :math:`(N, K)` consisting of
        :math:`K` measured attributes from :math:`N` independent observations.

    :param yerr: [optional]
        The errors on the data values. The errors are assumed to be normally
        distributed.

    :param quantum: [optional]
        The acceptable rounding-off quantum for the minimum message length
        in units of nits. Default is 0.1 nit.
    """

    parameter_names = ("mean", "sigma", "factor_scores", "factor_loads")

    def __init__(self, y, yerr=None, **kwargs):

        super(SingleLatentFactorEstimator, self).__init__(**kwargs)

        y = np.atleast_2d(y)
        N, K = y.shape
        if N < 1 or K < 1:
            raise ValueError("N and K must be at both positive integers")

        if yerr is not None:
            raise NotImplementedError(
                "perturbed data not available in single latent factor estimators yet")

        self._y, self._yerr = (y, yerr)

        self._mean, self._sigma, self._factor_scores, self._factor_loads \
            = self.estimate_parameters()

        return None


    @property
    def data(self):
        """
        Return the data values, :math:`y`.
        """
        return self._y


    @property
    def y(self):
        """
        Return the data values, :math:`y`.
        """
        return self._y


    @property
    def yerr(self):
        """
        Return the errors on the data values.
        """
        return self._yerr


    @property
    def mean(self):
        return self._mean

    @property
    def sigma(self):
        return self._sigma

    @property
    def factor_scores(self):
        return self._factor_scores

    @property
    def factor_loads(self):
        return self._factor_loads


    def estimate_parameters(self):
        r"""
        Return the initial estimates of parameters using minimum message length,
        under the assumption that the data are unperturbed.

        :returns:
            A four-length tuple containing: the means :math:`\mu_k`,
            the uncertainties :math:`\sigma_k`, the factor scores :math:`v_n`,
            and the factor loads :math:`a_k`.
        """

        N, K = self.y.shape
        mean = np.mean(self.y, axis=0)
        sigma = np.array([0.1, 0.2, 0.4])
        v = np.ones(N)/N
        a = np.ones(K)

        return (mean, sigma, v, a)
        

    def _iterate_parameters(self):
        r"""
        A generator that yields the best estimates for the model parameters,
        using minimum message length. See p.299 (Table 6.4) of Wallace (2005).

        """

        means = np.mean(self.y, axis=0)

        w_nk = self.y - self.mean
        y_nk = w_nk/self.sigma
        b_k = self.factor_loads/self.sigma
        b_squared = np.sum(b_k**2)
        #Y = 
        raise NotImplementedError


    @property
    def log_prior(self):
        r"""
        Return the natural logarithm of the prior.

        The priors on the means :math:`\mu_k` are assumed to be uniform over
        some bounded range, if given, and that region is expected to be large
        compared to :math:`a_k`.

        The prior on :math:`\sigma_k` has a density proportional to 
        :math:`1/\sigma_k` over some (given) finite range.

        For the prior on :math:`\underline{b}` (where :math:`b_k = a_k/\sigma_k`)
        we assume all directions in :math:`K`-space to be equally likely, and 
        a prior density in :math:`K`-space proportional to:

        .. math::

            p(b) = (1 + b^2)^{-(K + 1)/2}

        which expresses the expectation that in each dimension, :math:`a_k`
        will be of the same order as :math:`\sigma_k` but could be considerably
        larger. For :math:`b^2 < 1`, it is slowly varying. For :math:`b^2 \gg 1`,
        it leads to a density for :math:`b = |\underline{b}|` proportional to
        :math:`1/b^2`. The resulting prior :math:`p(a|\sigma)` is:

        .. math::

            p(a|\sigma) = (1 + b^2)^{-(K + 1)/2} \prod_{k}\frac{1}{\sigma_k}
        
        Excluding the bounds on :math:`\mu_k`, 
        the total logarithm of the prior is:
        # TODO

        .. math::

            \log{p(\theta)} = -2\sum_{k}\log{\sigma_k}
                              -\frac{1}{2}(K + 1)\log{1 + b^2}
        """
        return _log_prior(self.sigma, self.factor_loads)


    @property
    def log_fisher(self):
        r"""
        Return the natural logarithm of the Fisher information.

        If :math:`K \ll N`, then Wallace (2005, p. 298) shows that the Fisher 
        information is:

        .. math::

            F(\underline{\mu},\underline{\sigma},\underline{a},\underline{v})
                = (2N)^{K}(Nv^2 - S^2)^{K}(1 + b^2)^{N-2} / \prod_{k}\sigma_{k}^6

        where :math:`S = \sum_{n}v_n` and :math:`b^2 = \underline{b}^2 = \sum_{k}b_k^2`
        and :math:`b_k = a_k/\sigma_k`. Thus the natural logarithm of the
        Fisher information is:

        .. math::

            \log{F} = K\left[\log{(2N)} + \log{(Nv^2 - S^2)}\right] 
                    + (N - 2)\log{(1 + b^2)}
                    - 6\sum_{k}\log{\sigma_k}
        """
        return _log_fisher(self.sigma, self.factor_scores, self.factor_loads)


    @property
    def log_likelihood(self):
        r"""
        Return the natural log likelihood of the data.

        From Section 6.9 of Wallace (2005), the *negative* log likelihood is:

        .. math::

            -\log{f(y|\theta)} = \frac{1}{2}KN\log(2\pi)
                               + N\sum_{k}\log{\sigma_k}
                               + \frac{1}{2}\sum_{n}\sum_{k}(y_{nk} - \mu_{k} 
                                   - v_{n}a_{k})^2/\sigma_{k}^2
        

        """
        return _log_likelihood(self.y, self.mean, self.sigma, 
            self.factor_scores, self.factor_loads)
