
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

        # Excuse the terminology problem here, but means here refers to the 
        # single mean for the entire sample, not the cluster means!
        # TODO REVISIT TERMINOLOGY
        means = np.mean(data, axis=0)
        
        w = data - means

        # Set the initial factor scores by their location along the principal
        # eigenvetor, scaled by the square root
        U, S, V = np.linalg.svd(w.T)
        factor_scores = (V[0] * S[0]**0.5).reshape((N, -1))
        factor_loads = np.mean(w/factor_scores, axis=0).reshape((-1, D))
        specific_variances = np.var(
            w - np.dot(factor_scores, factor_loads), axis=0)

        # Do initial clustering on the factor scores.
        responsibility = np.zeros((N, self.num_components))
           
        if self.initialization_method == "kmeans++":
            # Assign cluster memberships using k-means++.
            kmeans = cluster.KMeans(
                n_clusters=self.num_components, n_init=1, **kwargs)
            assignments = kmeans.fit(factor_scores).labels_
            
        elif self.initialization_method == "random":
            # Randomly assign points.
            assignments = np.random.randint(0, self.num_components, size=N)

        responsibility[np.arange(N), assignments] = 1.0

        # Calculate the weights from the responsibility initialisation.
        effective_membership = np.sum(responsibility, axis=0)
        weights = (effective_membership + 0.5)/(N + D/2.0)

        # We need to initialise the factor scores to be of K clusters.
        # TODO: Revisit this whether this should be effective_membership > 1
        cluster_factor_scores = np.dot(factor_scores.T, responsibility) \
                              / effective_membership

        logger.warn("DANGER WILL ROBINSON")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(data.T[0], data.T[1], facecolor="#666666")

        bar = means + np.dot(factor_scores, factor_loads)
        ax.scatter(bar.T[0], bar.T[1], facecolor="g", alpha=0.5)

        colors = "br"
        for k in range(2):
            foo = means + np.dot(cluster_factor_scores.T[k], factor_loads)

            ax.scatter(foo.T[0], foo.T[1], facecolor=colors[k])

        # Set the parameters.
        return self.set_parameters(means=means, weights=weights,
            factor_loads=factor_loads, 
            cluster_factor_scores=cluster_factor_scores, 
            specific_variances=specific_variances)

    

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

        # Get the current factor scores, given the responsibility matrix.
        #w = (data - self.means)/self.factor_loads
        #factor_scores = np.sum(responsibility.T * w, axis=1).reshape((N, -1))
            
        factor_scores = np.dot(responsibility.T, self.cluster_factor_scores.T)

        # Update the factor loads.
        factor_loads = np.mean(
            (data - self.means)/factor_scores, axis=0).reshape((-1, D))

        sigmas = np.ones((90, 2)) * self.specific_variances**0.5
        sigmas = sigmas.flatten()

        def y(x, *factor_loads):
            factor_loads = np.array(factor_loads).reshape((1, 2))
            return (self.means + np.dot(factor_scores, factor_loads)).flatten()

        import scipy.optimize as op
        result = op.curve_fit(y, None, data.flatten(), sigma=sigmas,
            p0=factor_loads.flatten())

        factor_loads = result[0].reshape((1, 2))


        def y2(x, *cluster_factor_scores):
            factor_scores = np.atleast_2d(cluster_factor_scores)
            factor_scores = np.sum(responsibility.T * factor_scores, axis=1).reshape((N, -1))

            return (self.means + np.dot(factor_scores, factor_loads)).flatten()

        result2 = op.curve_fit(y2, None, data.flatten(), sigma=sigmas,
            p0=self.cluster_factor_scores)


        # Calculate the cluster factor scores.
        #cluster_factor_scores = np.dot(responsibility, factor_scores).T \
        #                      / effective_membership

        #means = np.zeros((K, D))
        #for k in range(K):
        #    means[k] = np.sum(responsibility[k] * data.T, axis=1) \
        #             / denominator[k]

        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(self.factor_scores.flatten(), factor_scores.flatten())

        limits = np.array([ax.get_xlim(), ax.get_ylim()])
        limits = (np.min(limits), np.max(limits))

        ax.plot(limits, limits)
        ax.set_xlim(limits)
        ax.set_ylim(limits)

        raise a
        """
        
        #logger.debug("_aecm_step_1 means {}".format(means))
        #logger.debug("_aecm_step_1 weights {}".format(weights))

        cluster_factor_scores = np.atleast_2d(result2[0])
        factor_scores = np.sum(responsibility.T * cluster_factor_scores, axis=1).reshape((N, -1))

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(data.T[0], data.T[1], facecolor="#666666")

        bar = self.means + np.dot(factor_scores, factor_loads)
        ax.scatter(bar.T[0], bar.T[1], facecolor="g", alpha=0.5)

        colors = "br"
        for k in range(2):
            foo = self.means + np.dot(cluster_factor_scores.T[k], factor_loads)

            ax.scatter(foo.T[0], foo.T[1], facecolor=colors[k])

        raise a

        return self.set_parameters(
            weights=weights,
            factor_loads=factor_loads,
            cluster_factor_scores=np.atleast_2d(result2[0]))


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
        

        '''
        K, N = responsibility.shape
        N, D = data.shape
        
        for k in range(K):
            residual = data - self.means

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
            raise a
        '''

        M, N = responsibility.shape
        N, K = data.shape

        # Subtract the means, weighted by responsibilities.
        #w_means = np.dot(responsibility.T, self.means)
        w = data - self.means
        
        b = (self.factor_loads/np.sqrt(self.specific_variances)).flatten()

        for iteration in range(self.max_em_iterations):

            #logger.debug("_aecm_step_2: iteration {}".format(iteration))

            # The iteration scheme is from Wallace & Freeman (1992)
            if (N - 1) * np.sum(b**2) <= K:
                b = np.zeros(K)

            specific_variances = np.sum(w**2, axis=0) \
                               / ((N - 1) * (1 + b**2))
            specific_sigmas = np.sqrt(np.atleast_2d(specific_variances))

            b_sq = np.sum(b**2)
            if b_sq == 0:
                raise NotImplementedError("a latent factor is not justified")
                
            # Note: Step (c) of Wallace & Freeman (1992) suggest to compute
            #       Y as Y_{kj} = \frac{V_{kj}}{\sigma_{k}\sigma_{j}}, where
            #       V_{kj} is the kj entry of the correlation matrix, but that
            #       didn't seem to work,...
            Y = np.dot(w.T, w) / np.dot(specific_sigmas.T, specific_sigmas)

            new_b = (np.dot(Y, b) * (1 - K/(N - 1) * b_sq)) \
                  / ((N - 1) * (1 + b_sq))

            change = np.sum(np.abs(b - new_b))

            #logger.debug("_aecm_step_2: {} {} {}".format(iteration, change, b))

            if not np.isfinite(change):
                raise a

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
            raise a

        specific_sigmas = np.sqrt(np.diag(np.dot(w.T, w)) / ((b*b + 1.0) * (N - 1)))
        factor_loads = specific_sigmas * b
        scaled_y = w/specific_sigmas

        b_sq = np.sum(b**2)
        factor_scores = np.dot(scaled_y, b) * (1 - K/(N - 1) * b_sq)/(1 + b_sq)
        specific_variances = specific_sigmas**2
        
        factor_loads = factor_loads.reshape((1, -1))
        factor_scores = factor_scores.reshape((-1, 1))
        specific_variances = specific_variances.reshape((1, -1))

        effective_membership = responsibility.sum(axis=1)
        # TODO: make this 1-N

        cluster_factor_scores = np.dot(responsibility, factor_scores).T \
                              / effective_membership

        return self.set_parameters(
            factor_loads=factor_loads,
            cluster_factor_scores=cluster_factor_scores,
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