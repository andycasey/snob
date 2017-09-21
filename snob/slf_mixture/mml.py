
"""
An estimator to model data using a fixed number of multivariate Gaussian 
distributions with a single multivariate latent factor, using minimum message
length as the objective function.
"""

__all__ = ["MMLMixtureModel"]

import logging
import numpy as np
import scipy
from sklearn import (cluster, decomposition)

from .base import BaseMixtureModel

logger = logging.getLogger("snob")


class MMLMixtureModel(BaseMixtureModel):


    def _initialize_parameters(self, data, **kwargs):
        r"""
        Initialize the model parameters, given the data.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        # Estimate the latent factor first.
        N, K = data.shape

        means = np.mean(data, axis=0)
        w = data - means
        correlation_matrix = np.corrcoef(w.T)

        U, S, V = np.linalg.svd(correlation_matrix)


        if kwargs.get("randomly_initialize_factor_loads", False):
            b = np.random.uniform(size=(K, )) - 0.5
        else:

            # Set the initial values for b as the principal eigenvector, scaled by
            # the square-root of the principal eigenvalue
            b = V[0] * S[0]**0.5



        specific_sigmas = np.sqrt(np.diag(np.dot(w.T, w)) \
                    / ((b*b + 1.0) * (N - 1)))
        factor_loads = specific_sigmas * b
        y = (data - means)/specific_sigmas

        raise a

        b_sq = np.sum(b**2)
        factor_scores = np.dot(y, b) * (1 - K/(N - 1) * b_sq)/(1 + b_sq)
        specific_variances = specific_sigmas**2
        specific_variances = specific_variances.reshape((1, -1))

        # Subtract off the factor loads and factor scores, then estimate the
        # cluster properties.
        factor_scores = np.atleast_2d(factor_scores).reshape((-1, 1))
        factor_loads = np.atleast_2d(factor_loads).reshape((1, -1))

        y_nlf = data - np.dot(factor_scores, factor_loads)

        if self.initialization_method == "kmeans++":

            raise a
            # Do k-means on the data without the latent factor.
            responsibility = np.zeros((N, self.num_mixtures))
            kmeans = cluster.KMeans(
                n_clusters=self.num_mixtures, n_init=1, **kwargs)
            labels = kmeans.fit(y_nlf).labels_
            responsibility[np.arange(N), labels] = 1

        elif self.initialization_method == "random":

            # Randomly assign points.
            responsibility = np.random.randint(0, self.num_mixtures, size=N)
            responsibility /= responsibility.sum(axis=1)[:, np.newaxis]

        else:
            raise ValueError("Unknown initialization method: {}".format(
                self.initialization_method))

        # Calculate the weights from the responsibility initialisation.
        effective_membership = np.sum(responsibility, axis=0)
        weights = (effective_membership + 0.5)/(N + K/2.0)

        # Set the parameters.
        return self.set_parameters(means=means, weights=weights,
            factor_loads=factor_loads, factor_scores=factor_scores, 
            specific_variances=specific_variances)



    def message_length(self, y, yerr=0.001, full_output=False):
        r"""
        Return the approximate message length of the model and the data,
        in units of bits.

        # TODO: Document in full, the equations used here for each message
        #       component.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :param yerr: [optional]
            The 1:math:`\sigma` uncertainties on the data values :math:`y`.
            This can be given as a :math:`N\times{}D` array, or as a float
            value for all observations in all dimensions (default: ``0.001``).

        :param full_output: [optional]
            If ``True``, the total message length is returned, as well as a
            dictionary with the message lengths of the individual components.

        :returns:
            The total message length in units of bits.
        """

        raise NotImplementedError("not checked yet")

        yerr = np.array(yerr)
        if yerr.size == 1:
            yerr = yerr * np.ones_like(y)
        elif yerr.shape != y.shape:
            raise ValueError("shape mismatch")

        log_prob_norm, log_responsibility = self._expectation(y)

        responsibility = np.exp(log_responsibility)

        N, M = responsibility.shape
        N, D = y.shape

        I = dict()

        # I(M) = M\log{2} + constant # [nats]
        I["I_m"] = M * np.log(2)

        # I(w) = \frac{(M - 1)}{2}\log{N} - \frac{1}{2}\sum_{j=1}^{M}\log{w_j} - \log{(M - 1)!} # [nats]
        # Recall: \log{(M-1)!} = \log{|\Gamma(M)|}
        I["I_w"] = (M - 1) / 2.0 * np.log(N) \
                 - 0.5 * np.sum(np.log(self.weights)) \
                 - scipy.special.gammaln(M)

        if D == 1:
            raise NotImplementedError

        else:
            raise a
            log_det_cov = -2 * _compute_log_det_cholesky(
                self.precision_cholesky, self.covariance_type, D)

            # \frac{1}{2}\sum_{j=1}^{M}\log{|F(\theta_j)|} = \frac{1}{2}D(D + 3)N_{eff} - D * np.log(2) - (D + 2)\log{|C|}
           
            I["I_F"] = 0.5 * (
                0.5 * D * (D + 3) * np.log(np.sum(responsibility, axis=0)) \
              - D * np.log(2) \
              - (D - 2) * log_det_cov)
            
        I["I_h"] = 0 # TODO: priors

        I["I_l"] = -np.sum(log_prob_norm) - np.sum(np.log(yerr)) 

        if self.covariance_type == "full":
            N_free_params = 0.5 * D * (D + 3) * M + (M - 1)
        else:
            raise NotImplementedError

        I["lattice"] = 0.5 * N_free_params * log_kappa(N_free_params) 
        I["constant"] = 0.5 * N_free_params

        I_total = np.hstack(I.values()).sum() / np.log(2) # [bits]

        if not full_output:
            return I_total

        for key in I.keys():
            I[key] /= np.log(2) # [bits]

        return (I_total, I)




def log_kappa(D):
    r"""
    Return an approximation of the logarithm of the lattice constant 
    :math:`\kappa_D` using the relationship:

    .. math::

        I_1(x) \approx \frac{\log{(D\pi)}}{D} - \log{2\pi} - 1

    where :math:`D` is the number of dimensions.

    :param D:
        The number of dimensions.

    :returns:
        The approximate logarithm of the lattice constant.
    """
    return np.log(D * np.pi)/D - np.log(2 * np.pi) - 1