
""" A Gaussian estimator for use with minimum message length. """

__all__ = ["GaussianEstimator"]

import logging
import numpy as np

logger = logging.getLogger(__name__)

from .estimator import Estimator

class GaussianEstimator(Estimator):

    """
    An estimator to model data from a Gaussian distribution,
    using quadratic approximations to strict minimum message length.

    See Section 5.6 of Wallace (2005) for more details.
    
    :param y:
        The data values.

    :param quantum: [optional]
        The precision of the data values `y`. This can be specified as a single
        value, or as an array with the same size as `y`.

    :param mean_bounds: [optional]
        The lower and upper bounds for the mean value.

    :param sigma_upper_bound: [optional]
        The upper bound for the sigma value.
    """

    parameter_names = ("mean", "sigma")

    def __init__(self, y, quantum=1e-6, mean_bounds=None, 
        sigma_upper_bound=None, **kwargs):

        self._bounds = dict(
            mean=mean_bounds, 
            sigma=(np.min(quantum), sigma_upper_bound))

        y = np.array(y).flatten()
        quantum = np.array([quantum]).flatten()

        if quantum.size == 1:
            quantum = np.repeat(quantum, y.size)

        elif quantum.size != y.size:
            raise ValueError("quantum value must be a single value, "
                             "or an array the same size as y")

        self._quantum = quantum

        super(GaussianEstimator, self).__init__(y, **kwargs)

        # We can set the parameter values, since they are analytic.
        self._set_parameter_values()
        return None


    def estimate_parameters(self):
        """
        Return the MML estimates.

        :returns:
            A two-length tuple containing the mean and standard deviation.
        """

        mu = np.mean(self.data)
        sigma = np.sqrt(
            np.sum((self.data - np.mean(self.data))**2)/(self.data.size - 1))

        return (mu, sigma)


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


    @property
    def log_prior(self):
        """
        Return the log likelihood of the prior.

        Here we assume a scale-invariant, :math:`1/\sigma` prior on 
        :math:`\sigma`, which corresponds to a uniform prior on
        :math:`\log\sigma`, and equivalently, a :math:`1/\sigma^2` prior on
        :math:`1/\sigma^2`.
        """

        lp = np.log(self.sigma)
        if self.bounds.get("mean", None) is not None:
            lp += np.ptp(self.bounds["mean"])

        if self.bounds.get("sigma", None) is not None:
            lower, upper = self.bounds["sigma"]
            if upper is not None: 
                lp += np.log(upper) - np.log(lower)

        return lp


    @property
    def log_fisher(self):
        """
        Return the log likelihood of the Fisher information matrix.

        The Fisher information for a Gaussian is given by:

        .. math::

            F(\mu, \sigma^2) = \frac{N}{\sigma^2} \times \frac{N}{2(\sigma^2)^2}


        However with respect to :math:`(\mu, \sigma)`, the Fisher information
        is (Section 5.6 of Wallace, 2005):

        .. math::

            F(\mu, \sigma) = \frac{2N^2}{\sigma^4}
        """

        return   np.log(2) \
               + 2 * np.log(self.data.size) \
               - 4 * np.log(self.sigma)


    @property
    def log_data(self):
        """
        Return the log likelihood of the data.
        
        From Section 3.3.1 of Wallace (2005), the *negative* log likelihood is
        given by:

        .. math::

            -\log{f(x|\theta)} = \frac{N}{2}\log{2\pi} 
                               + N\log{\sigma} 
                               + \frac{N(\bar{y} - \mu)^2 
                                     + \sum_{n}(y_n - \bar{y})^2}{2\sigma^2} 

        And there is an additional :math:`-N\log{\epsilon}` term to deal with
        the quantum precision of the data :math:`y`.
        """

        # Note that the term:
        # (N * (\bar{y} - \mu)^2)/(2\sigma^2)
        # or:
        # self.data.size * (np.mean(self.data) - self.mean)**2
        # is excluded because it will always be zero.

        return -(  0.5 * self.data.size * np.log(2 * np.pi)
                 + self.data.size * np.log(self.sigma)
                 + np.sum((self.data - self.mean)**2)/(2*self.sigma**2)
                 - np.sum(np.log(self.quantum))
                 )