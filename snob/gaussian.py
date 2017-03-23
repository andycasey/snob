
""" A Gaussian estimator for use with minimum message length. """

__all__ = ["GaussianEstimator"]

import logging
import numpy as np
import scipy.optimize as op

logger = logging.getLogger(__name__)

from . import estimator


def _log_prior(sigma, bounds, quantum):
    r"""
    Return the log likelihood of the prior.

    Here we assume a scale-invariant, :math:`1/\sigma` prior on 
    :math:`\sigma`, which corresponds to a uniform prior on
    :math:`\log\sigma`, and equivalently, a :math:`1/\sigma^2` prior on
    :math:`1/\sigma^2`.
    """

    lp = np.log(sigma)
    mean_bounds, sigma_bounds = bounds
    if None not in mean_bounds:
        lp += np.ptp(mean_bounds)

    if sigma_bounds[1] is not None:
        lp += np.log(sigma_bounds[1]) - np.log(quantum)

    return lp


def _log_fisher(sigma, yerr):
    r"""
    Return the natural logarithm of the Fisher information.

    For unperturbed data, the Fisher information is (Section 5.6 of
    Wallace, 2005):

    .. math::

        F(\mu, \sigma) = \frac{2N^2}{\sigma^4}


    However for perturbed data:

    .. math::

        F(\mu, \sigma) = \frac{2}{\sigma^4}\left(\sum_{n}w_n\right)\left(\sum_{n}w_n^2\right)

    Where :math:`w_n = \sigma^2/(\sigma^2 + \epsilon_n)` and 
    :math:`\epsilon_n` is the normally-distributed error on datum
    :math:`y_n`. The unperturbed and perturbed cases are equivalent when 
    :math:`\epsilon_n = 0`.

    :param sigma:
        The standard deviation :math:`\sigma` of the Gaussian distribution.

    :param yerr:
        The normally-distributed 1-sigma uncertainties on the data :math:`y`.
    """

    weights = sigma**2/(sigma**2 + yerr**2)
    return np.sum([
        +np.log(2),
        -4*np.log(sigma),
        +np.log(np.sum(weights)),
        +np.log(np.sum(weights**2))
    ])


def _log_data(mean, sigma, y, quantum):
    r"""
    Return the log likelihood of the data.
    
    From Section 5.6 of Wallace (2005), the *negative* log likelihood is
    given by:

    .. math::


        -\log{f(x|\mu,\sigma)} = (N/2)\log{(2\pi)} 
                               + N\log\sigma
                               - N\log\epsilon
                               + \sum_n(y_n - \mu)^2/(2\sigma^2)

    Where :math:`\epsilon` represents the quantum of the data, :math:`y`.
    """
    return -np.sum([
        +(y.size/2.0)*np.log(2*np.pi),
        +y.size*np.log(sigma),
        -np.sum(np.log(quantum)),
        +np.sum((y - mean)**2)/(2*sigma**2)
    ])


def _message_length(parameters, y, yerr, bounds, quantum):
    """
    Return the approximate message length of the Gaussian estimator, given the
    parameters.

    :param parameters:
        A two-length list-like object containing the Gaussian mean and sigma.

    :param y:
        The data values :math:`y`.

    :param yerr:
        The normally-distributed sigma uncertainties on :math:`y`.

    :param bounds:
        A list containing two-length tuples that contain the lower and upper
        bounds for all parameter names. If no edge bound exists for a parameter
        then `None` is provided.

    :param quantum:
        The acceptable rounding-off quantum for the minimum message length
        in units of nits. Default is 0.1 nit.
    """

    mean, sigma = parameters

    return np.sum([
        -_log_prior(sigma, bounds, quantum),
        +0.5*_log_fisher(sigma, yerr),
        -_log_data(mean, sigma, y, quantum),
        +estimator.log_kappa(2)
    ])


class GaussianEstimator(estimator.Estimator):

    """
    An estimator to model data from a one dimensional Gaussian distribution,
    using quadratic approximations to strict minimum message length. This
    estimator can handle unperturbed data (e.g., no errors), as well as
    perturbed data. See Sections 5.6 and 6.3 of Wallace (2005).

    :param y:
        The data values.

    :param yerr: [optional]
        The errors on the data values. The errors are assumed to be normally
        distributed.

    :param mean_bounds: [optional]
        The lower and upper bounds for the mean value.

    :param sigma_upper_bound: [optional]
        The upper bound for the sigma value.

    :param quantum: [optional]
        The acceptable rounding-off quantum for the minimum message length
        in units of nits. Default is 0.1 nit.
    """

    parameter_names = ("mean", "sigma")

    def __init__(self, y, yerr=None, mean_bounds=None, sigma_upper_bound=None,
        **kwargs):

        super(GaussianEstimator, self).__init__(**kwargs)

        y = np.array(y).flatten()

        if yerr is None:
            yerr = np.zeros(y.size)
        else:
            yerr = np.array(yerr).flatten()
            if yerr.size != y.size:
                raise ValueError("size mis-match between y and yerr "\
                                 "({} != {})".format(y.size, yerr.size))

        if not np.all(yerr >= 0):
            raise ValueError("all yerr values must be positive")

        # Check the mean bounds prior.
        if mean_bounds is None:
            mean_bounds = (None, None)

        else:
            # Check that we have a proper prior.
            if len(mean_bounds) != 2:
                raise ValueError(
                    "mean_bounds must be None or a two-length tuple")

            if list(mean_bounds).count(None) == 1:
                raise ValueError("unbounded prior given in mean_bounds")

            elif None not in mean_bounds:
                mean_bounds = np.sort(mean_bounds)

            mean_bounds = tuple(mean_bounds)


        self._y, self._yerr = (y, yerr)
        self._bounds = [
            mean_bounds,
            (0, sigma_upper_bound)
        ]
        
        # Generate initial estimates for the parameters.
        self._mean, self._sigma = self.estimate_parameters()
        return None


    def estimate_parameters(self):
        """
        Return the parameter estimates using minimum message length under the
        assumption that the data are unperturbed.        

        :returns:
            A two-length tuple containing the mean and standard deviation.
        """

        mean = np.mean(self.y)
        sigma = np.sqrt(np.sum((self.y - mean)**2)/(self.y.size - 1))
        return (mean, sigma)


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
    def bounds(self):
        """
        Return the bounds on the parameter names.
        """
        return self._bounds

    
    @property
    def mean(self):
        """
        Return the minumum message length estimate of the mean.
        """
        return self._mean


    @property
    def sigma(self):
        """
        Return the minimum message length estimate of the standard deviation.
        """
        return self._sigma


    def optimize(self, x0=None, **kwargs):
        """
        Optimize the parameters by minimising the message length.

        :param x0: [optional]
            The initial values for the parameters. If `None` is given, the
            current model attributes will be used.

        :returns:
            A two-length tuple containing the optimised parameters,
            and a dictionary containing relevant metadata. The model attributes
            will be updated automatically to the optimized values.
        """

        args = (self.y, self.yerr, self.bounds, self.quantum)

        if x0 is None:
            x0 = np.array([getattr(self, pn) for pn in self.parameter_names])

        kwds = dict(factr=10, disp=0, iprint=-1, fprime=None, callback=None,
            approx_grad=True, bounds=self.bounds)
        kwds.update(kwargs)

        x, f, d = op.fmin_l_bfgs_b(_message_length, x0, args=args, **kwds)

        d.update(dict(func=f, x=x, x0=x0, kwds=kwds))
        if d.get("warnflag", 0) > 0:
            task = "too many function evaluations or too many iterations" \
                    if d["warnflag"] == 1 else d["task"]
            logger.warn("op.fmin_l_bfgs_b returned warning: {}".format(task))

        self._mean, self._sigma = x

        return (x, d)


    @property
    def log_prior(self):
        r"""
        Return the log likelihood of the prior.

        Here we assume a scale-invariant, :math:`1/\sigma` prior on 
        :math:`\sigma`, which corresponds to a uniform prior on
        :math:`\log\sigma`, and equivalently, a :math:`1/\sigma^2` prior on
        :math:`1/\sigma^2`.
        """
        return _log_prior(self.sigma, self.bounds, self.quantum)


    @property
    def log_fisher(self):
        r"""
        Return the natural logarithm of the Fisher information.

        For unperturbed data, the Fisher information is (Section 5.6 of
        Wallace, 2005):

        .. math::

            F(\mu, \sigma) = \frac{2N^2}{\sigma^4}


        However for perturbed data:

        .. math::

            F(\mu, \sigma) = \frac{2}{\sigma^4}\left(\sum_{n}w_n\right)\left(\sum_{n}w_n^2\right)

        Where :math:`w_n = \sigma^2/(\sigma^2 + \epsilon_n)` and 
        :math:`\epsilon_n` is the normally-distributed error on datum
        :math:`y_n`. The unperturbed and perturbed cases are equivalent when 
        :math:`\epsilon_n = 0`.
        """
        return _log_fisher(self.sigma, self.yerr)


    @property
    def log_data(self):
        r"""
        Return the log likelihood of the data.
        
        From Section 5.6 of Wallace (2005), the *negative* log likelihood is
        given by:

        .. math::


            -\log{f(x|\mu,\sigma)} = (N/2)\log{(2\pi)} 
                                   + N\log\sigma
                                   - N\log\epsilon
                                   + \sum_n(y_n - \mu)^2/(2\sigma^2)

        Where :math:`\epsilon` represents the quantum of the data, :math:`y`.
        """
        return _log_data(self.mean, self.sigma, self.y, self.quantum)