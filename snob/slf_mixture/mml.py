
"""
An estimator to model data using a fixed number of multivariate Gaussian 
distributions with a single multivariate latent factor, using minimum message
length as the objective function.
"""

__all__ = ["MMLMixtureModel"]

import logging
import numpy as np
import scipy
from sklearn import cluster

from .base import BaseMixtureModel

logger = logging.getLogger("snob")


class MMLMixtureModel(BaseMixtureModel):


    def initialize_parameters(self, data, **kwargs):
        r"""
        Initialize the model parameters, given the data.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        # Estimate the latent factor first.
        N, D = data.shape
        K = self.num_components

        # Excuse the terminology problem here, but means here refers to the 
        # single mean for the entire sample, not the cluster means!
        # TODO REVISIT TERMINOLOGY
        self.set_parameters(means=np.mean(data, axis=0))
        
        w = data - self.means
        V = np.dot(w.T, w)
        
        # Set \beta as the dominant eigenvector of the correlation matrix with
        # length given by setting b^2 equal to the dominant eigenvalue.
        _, s, v = np.linalg.svd(np.corrcoef(data.T))
        beta = v[0] * np.sqrt(s[0])

        # Iterate as per Wallace & Freeman first.
        for i in range(self.max_em_iterations):
            beta, specific_variances, change = self._iterate_wf90(beta, V, N, D)

            logger.debug("initialize_parameters: {} {}".format(i, change))
            if change is None or change < self.threshold: break
        

        factor_loads = (np.sqrt(specific_variances) * beta).reshape((1, D))
        b_sq = np.sum(beta**2)
        #factor_scores = np.dot(w/specific_sigmas, beta) \
        #              * ((1.0 - D/((N - 1.0) * b_sq)) / (1 + b_sq))
        factor_scores = np.dot(data, beta) \
                      * ((1.0 - D/(N - 1.0) * b_sq) / (1 + b_sq))
        factor_scores = np.atleast_2d(factor_scores).T

        # Do initial clustering on the factor scores.
        responsibility = np.zeros((K, N))
           
        if self.initialization_method == "kmeans++":
            # Assign cluster memberships using k-means++.
            kmeans = cluster.KMeans(n_clusters=K, n_init=1, **kwargs)
            assignments = kmeans.fit(factor_scores).labels_
            
        elif self.initialization_method == "random":
            # Randomly assign points.
            assignments = np.random.randint(0, K, size=N)

        responsibility[assignments, np.arange(N)] = 1.0

        # Calculate the weights from the responsibility initialisation.
        effective_membership = np.sum(responsibility, axis=1)
        weights = (effective_membership + 0.5)/(N + D/2.0)

        # We need to initialise the factor scores to be of K clusters.
        # TODO: Revisit this whether this should be effective_membership > 1
        cluster_factor_score_means = (np.dot(factor_scores.T, responsibility.T) \
                                   / (effective_membership - 1))
        cluster_factor_score_means = cluster_factor_score_means.T
        
        covs = np.zeros((K, D, D))

        # Initialise the sigmas
        for k in range(K):

            factor_scores = cluster_factor_score_means[k] * np.ones((N, 1))
            residual = (data - self.means - np.dot(factor_scores, factor_loads))
            covs[k] = np.dot(responsibility[k] * residual.T, residual) \
                    / (effective_membership[k] - 1)
            
        # let's estimate the intrinsic specific variances
        variances = np.array([np.diag(cov) for cov in covs])

        specific_variances = np.min(variances, axis=0)/2.0
        cluster_variances = variances - specific_variances

        # TODO: no idea if this variance stuff is right,.....
        cluster_factor_score_variances \
            = np.mean(cluster_variances/(factor_loads**2), axis=1).reshape((K, 1))

        # Set the parameters.
        self.set_parameters(
            weights=weights,
            factor_loads=factor_loads, 
            specific_variances=specific_variances,
            cluster_factor_score_means=cluster_factor_score_means,
            cluster_factor_score_variances=cluster_factor_score_variances)

    
    def _aecm_step_1(self, data, responsibility):
        r"""
        Perform the first conditional maximization step of the ECM algorithm,
        where we update the weights, and the factor loads and scores.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :param responsibility: 
            The responsibility matrix for all :math:`N` observations being
            partially assigned to each :math:`K` component.
        """

        logger.debug("_aecm_step_1")

        K, N = responsibility.shape
        N, D = data.shape

        # Update the weights by minimising the message length.
        effective_membership = np.sum(responsibility, axis=1)
        weights = (effective_membership + 0.5)/(N + K/2.0)

        return self.set_parameters(weights=weights)


    def _aecm_step_2(self, data, responsibility):
        r"""
        Perform the second conditional maximization step of the AECM algorithm,
        where we update the factor loads, factor scores and specific variances.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :param responsibility: 
            The responsibility matrix for all :math:`N` observations being
            partially assigned to each :math:`K` component.
        """
        
        # Do a single-step update like we did in the initialisation.
        # (If this doesn't work we are going straight to an optimiser!)
        N, D = data.shape        
        w = data - self.means
        V = np.dot(w.T, w)
        
        beta, specific_variances, l1_norm = _iterate_wallace_freeman_1990(
            (self.factor_loads * self.specific_variances**(-0.5)).T, N, D, V)
            
        logger.debug("_aecm_step_2 l1_norm: {}".format(l1_norm))

        specific_sigmas = specific_variances**0.5
        factor_loads = specific_sigmas * beta

        b_sq = np.sum(beta**2)
        factor_scores = np.dot(w/specific_sigmas, beta) \
                      * ((1.0 - D/((N - 1.0) * b_sq)) / (1 + b_sq))
        factor_scores = np.atleast_2d(factor_scores).T

        # Calculate cluster factor scores.
        effective_membership = np.sum(responsibility, axis=1)
        cluster_factor_scores = np.dot(factor_scores.T, responsibility.T) \
                              / (effective_membership - 1)

        return self.set_parameters(
            factor_loads=factor_loads,
            specific_variances=specific_variances,
            factor_scores=factor_scores,
            cluster_factor_scores=cluster_factor_scores)



    def message_length(self, y, yerr=0.001, full_output=False):
        return None
        


    def _iterate_wf90(self, beta, V, N, D):

        beta = beta.reshape((D, ))

        b_squared = np.sum(beta**2)

        lhs = (N - 1.0) * b_squared
        if lhs <= D:
            logger.warn("Setting \Beta_k = 0 as {} <= {}".format(lhs, D))
            beta, b_squared = (np.zeros(D), 0)

        # Compute specific variances according to:
        #   \sigma_k^2 = \frac{\sum_{n}w_{nk}^2}{(N-1)(1 + \Beta_k^2)}
        # (If \Beta = 0, exit)

        # Discrepancy between Wallace (2005) and Wallace and Freeman (1992) here
        # as to whether this should be \beta_k^2 or b^2
        specific_variances = np.diag(V) / ((N - 1.0) * (1.0 + beta**2))

        assert specific_variances.size == D

        if np.all(beta == 0):
            logger.warn("Exiting step (b) of scheme in initialisation")
            return (beta, specific_variances, None)

        # Compute Y using 
        #   Y_{kj} = V_{kj}/\sigma_{k}\sigma_{j}
        # which is Wallace & Freeman nomenclature. In our nomenclature
        # that is:
        #   Y_{ij} = V_{ij}/\sigma_{i}\sigma_{j}

        specific_sigmas = np.atleast_2d(specific_variances**0.5)
        Y = V / np.dot(specific_sigmas.T, specific_sigmas)

        # Compute an updated estimate of \beta as:
        #   \Beta_{new} = \frac{Y\Beta(1-K/(N-1)b^2)}{(N - 1)(1 + b^2)}
        # which is Wallace & Freeman nomenclature. In our nomenclature
        # that is:
        #   \Beta_{new} = \frac{Y\Beta(1-D/(N - 1)b^2)}{(N - 1)(1 + b^2)}

        # Where you will recall (in our nomenclature)
        #   b^2 = \sum_{D} b_d^2
        # And b_{d} = a_{d}/\sigma_{d}

        b_squared = np.sum(beta**2)
        beta_new = (np.dot(Y, beta) * (1.0 - D/(N - 1.0) * b_squared)) \
                 / ((N - 1.0) * (1.0 + b_squared))

        # If \Beta_new \approx \Beta then exit.
        l1_norm = np.abs(beta - beta_new).sum()

        if not np.isfinite(l1_norm):
            raise ValueError("l1_norm in _iterate_wallace_freeman_1990 is not finite")

        return (beta_new, specific_variances, l1_norm)



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