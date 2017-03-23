
""" An estimator for a single multivariate latent factor. """

__all__ = ["SingleLatentFactorEstimator"]

import logging
import numpy as np
from . import estimator

logger = logging.getLogger(__name__)


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
    return -np.sum([
        0.5*N*K*np.log(2*np.pi),
        N * np.sum(np.log(sigma)),
        0.5 * np.sum((y - means - v * a)**2 / sigma**2)
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
    b_squared = np.sum((a/sigma)**2)
    S = np.sum(v)
    return np.prod([ # for code legibility
        (2*N)**K,
        (N * v**2 - S**2)**K,
        (1 + b_squared)**(N - 2),
        np.prod(sigma**-6)
    ])


def _log_prior(*args, **kwargs):
    print("no prior")
    return 0


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



    def estimate_parameters(self):
        r"""
        Return the initial estimates of parameters using minimum message length,
        under the assumption that the data are unperturbed.

        :returns:
            A four-length tuple containing: the means :math:`\mu_k`,
            the uncertainties :math:`\sigma_k`, the factor scores :math:`v_n`,
            and the factor loads :math:`a_k`.
        """

        means = np.mean(self.y, axis=0)
        raise NotImplementedError


    @property
    def log_prior(self):
        return _log_prior()

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
        and :math:`b_k = a_k/\sigma_k`.
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

