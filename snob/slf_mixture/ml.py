
"""
An estimator to model data using a fixed number of multivariate Gaussian 
distributions with a single multivariate latent factor, using maximum
likelihood as the objective function.
"""

__all__ = ["MLMixtureModel"]

import logging
import numpy as np
import scipy
from sklearn import (cluster, decomposition)

from .base import BaseMixtureModel

logger = logging.getLogger("snob")


class MLMixtureModel(BaseMixtureModel):
    

    def initialize_parameters(self, y, **kwargs):

        N, D = y.shape

        # Initialize the factor load.
        fa = decomposition.FactorAnalysis(n_components=1, **kwargs)
        fa.fit(y)

        factor_load = fa.components_
        specific_variance = fa.noise_variance_

        b = factor_load/np.sqrt(specific_variance)
        b_sq = np.sum(b**2)
        factor_score = np.dot(y, b.T) * (1 - D/(N - 1) * b_sq)/(1 + b_sq)

        y_nlf = y - np.dot(factor_score, factor_load)

        if self.initialization_method == "kmeans++":

            K = self.num_components
            row_norms = cluster.k_means_.row_norms(y_nlf, squared=True)
            mean = cluster.k_means_._k_init(
                y_nlf, K, row_norms, 
                cluster.k_means_.check_random_state(None))

            distance = np.sum((y_nlf[:, :, None] - mean.T)**2, axis=1).T

            responsibility = np.zeros((K, N))
            responsibility[np.argmin(distance, axis=0), np.arange(N)] = 1.0

            weight = responsibility.sum(axis=1)/N

        else:
            raise NotImplementedError("random assignment not implemented yet"\
                                      " for max likelihood SLF mixture model")

        # Set the parameters.
        return self.set_parameters(means=mean, weights=weight,
            factor_loads=factor_load, factor_scores=factor_score, 
            specific_variances=specific_variance)



    def _aecm_step_1(self, data, responsibility):
        r"""
        Perform the first conditional maximization step of the ECM algorithm,
        where we update the weights and the means.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :param responsibility: 
            The responsibility matrix for all :math:`N` observations being
            partially assigned to each :math:`K` component.
        """

        # Update the estimate of the latent factor, after subtracting the
        # means from each cluster.
        M, N = responsibility.shape
        N, K = data.shape

        '''
        effective_membership = np.sum(responsibility, axis=1)
        denominator = effective_membership.copy()
        denominator[denominator > 1] = denominator[denominator > 1] - 1

        means = np.zeros((M, K))
        for m in range(M):
            means[m] = np.sum(responsibility[m] * data.T, axis=1) \
                     / denominator[m]

        weights = (effective_membership + 0.5)/(N + M/2.0)
        '''

        # This is the ML implementatin, not MML.
        effective_membership = np.sum(responsibility, axis=1)
        weights = effective_membership/N

        means = np.zeros((M, K))
        for m in range(M):
            means[m] = np.sum(responsibility[m] * data.T, axis=1) \
                     / effective_membership[m]

        return self.set_parameters(weights=weights, means=means)


    def _aecm_step_2(self, data, responsibility):
        r"""
        Perform the second conditional maximization step of the AECM algorithm,
        where we update the factor loads and factor scores.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :param responsibility: 
            The responsibility matrix for all :math:`N` observations being
            partially assigned to each :math:`K` component.
        """
        
        N, D = data.shape
        K, N = responsibility.shape

        for k in range(K):
            residual = data - self.means[k]


            gamma = (
                  np.dot(self.factor_loads, self.factor_loads.T) \
                + self.specific_variances)**(-1) * self.factor_loads

            omega = np.eye(D) - np.dot(gamma.T, self.factor_loads)

            V = np.dot(responsibility[k], np.dot(residual, residual.T)) \
              / np.sum(responsibility[k])
            V = np.atleast_2d(V).T

            Cinv = np.linalg.inv((
                np.dot(gamma, np.atleast_2d(np.dot(V, gamma).sum(axis=0)).T) + omega))
            B = np.dot(V, np.dot(gamma, Cinv))

            #V = self.weight * np.dot(residual, residual.T) \
            #  / np.sum(self.)



        M, N = responsibility.shape
        N, K = data.shape

        # Subtract the means, weighted by responsibilities.
        w_means = np.dot(responsibility.T, self.means)
        w = data - w_means
        
        # Get best current estimate of b
        assert self.factor_loads is not None
        assert self.specific_variances is not None

        b = (self.factor_loads/np.sqrt(self.specific_variances)).flatten()

        for iteration in range(self.max_em_iterations):

            # The iteration scheme is from Wallace & Freeman (1992)
            #if (N - 1) * np.sum(b**2) <= K:
            #    b = np.zeros(K)

            specific_variances = np.sum(w**2, axis=0) \
                               / ((N - 1) * (1 + b**2))
            specific_sigmas = np.sqrt(np.atleast_2d(specific_variances))

            b_sq = np.sum(b**2)
            #if b_sq == 0:
            #    raise NotImplementedError("a latent factor is not justified")
                
            # Note: Step (c) of Wallace & Freeman (1992) suggest to compute
            #       Y as Y_{kj} = \frac{V_{kj}}{\sigma_{k}\sigma_{j}}, where
            #       V_{kj} is the kj entry of the correlation matrix, but that
            #       didn't seem to work,...
            Y = np.dot(w.T, w) / np.dot(specific_sigmas.T, specific_sigmas)

            new_b = (np.dot(Y, b) * (1 - K/(N - 1) * b_sq)) \
                  / ((N - 1) * (1 + b_sq))

            change = np.sum(np.abs(b - new_b))
            if not np.isfinite(change):
                logger.warn(
                    "Non-finite change on iteration {}".format(iteration))
                b = 0.5 * (self.factor_loads/np.sqrt(self.specific_variances)).flatten()
                break

            b = new_b        

            logger.debug(
                "Iteration #{} on SLF, delta: {}".format(iteration, change))

            if self.threshold >= change:
                logger.debug(
                    "Tolerance achieved on iteration {}".format(iteration))
                break

        else:
            logger.warn("Scheme did not converge")
            # Iterative scheme didn't converge.
            b = 0.5 * (self.factor_loads/np.sqrt(self.specific_variances)).flatten()
            
        specific_sigmas = np.sqrt(np.diag(np.dot(w.T, w)) \
                    / ((b*b + 1.0) * (N - 1)))
        factor_loads = specific_sigmas * b
        scaled_y = (data - w_means)/specific_sigmas

        b_sq = np.sum(b**2)
        factor_scores = np.dot(scaled_y, b) * (1 - K/(N - 1) * b_sq)/(1 + b_sq)
        specific_variances = specific_sigmas**2
        
        factor_loads = factor_loads.reshape((1, -1))
        factor_scores = factor_scores.reshape((-1, 1))
        specific_variances = specific_variances.reshape((1, -1))

        return self.set_parameters(
            factor_loads=factor_loads,
            factor_scores=factor_scores,
            specific_variances=specific_variances)