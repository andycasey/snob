
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

    @property
    def approximate_factor_scores(self):
        """
        Return approximate factor scores for each object, based on the
        responsibility matrix.
        """

        return self._factor_scores
        return np.dot(self._responsibility.T, self.cluster_factor_scores.T)


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
        means = np.zeros(D)
        
        w = data - means
        V = np.dot(w.T, w)
        
        # Set \beta as the dominant eigenvector of the correlation matrix with
        # length given by setting b^2 equal to the dominant eigenvalue.
        _, s, v = np.linalg.svd(np.corrcoef(data.T))
        beta = v[0] * np.sqrt(s[0])

        # Iterate as per Wallace & Freeman first.
        for iteration in range(self.max_em_iterations):

            beta, specific_variances, l1_norm = _iterate_wallace_freeman_1990(
                beta, N, D, V)

            logger.debug("initialize_parameters: {} {}".format(iteration, l1_norm))

            # HACK TODO MAGIC:
            if l1_norm is None or l1_norm < 1e-10:
                break

        specific_sigmas = specific_variances**0.5
        factor_loads = specific_sigmas * beta

        b_sq = np.sum(beta**2)
        factor_scores = np.dot(w/specific_sigmas, beta) \
                      * ((1.0 - D/((N - 1.0) * b_sq)) / (1 + b_sq))
        factor_scores = np.atleast_2d(factor_scores).T

        # Do initial clustering on the factor scores.
        responsibility = np.zeros((self.num_components, N))
           
        if self.initialization_method == "kmeans++":
            # Assign cluster memberships using k-means++.
            kmeans = cluster.KMeans(
                n_clusters=self.num_components, n_init=1, **kwargs)
            assignments = kmeans.fit(factor_scores).labels_
            
        elif self.initialization_method == "random":
            # Randomly assign points.
            assignments = np.random.randint(0, self.num_components, size=N)

        responsibility[assignments, np.arange(N)] = 1.0

        # Calculate the weights from the responsibility initialisation.
        effective_membership = np.sum(responsibility, axis=1)
        weights = (effective_membership + 0.5)/(N + D/2.0)

        # We need to initialise the factor scores to be of K clusters.
        # TODO: Revisit this whether this should be effective_membership > 1
        cluster_factor_scores = np.dot(factor_scores.T, responsibility.T) \
                              / (effective_membership - 1)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(data.T[0], data.T[1], c=responsibility[0], zorder=-1)

        colors = "br"
        for i in range(self.num_components):
            y = means + np.dot(cluster_factor_scores.flatten()[i], factor_loads)
            ax.scatter(y.T[0], y.T[1], c=colors[i])


        # Set the parameters.
        return self.set_parameters(means=means, weights=weights,
            factor_loads=factor_loads, 
            factor_scores=factor_scores,
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
        
        yerr = np.array(yerr)
        if yerr.size == 1:
            yerr = yerr * np.ones_like(y)
        elif yerr.shape != y.shape:
            raise ValueError("shape mismatch")

        responsibility, log_likelihood = self._expectation(y)

        K, N = responsibility.shape
        N, D = y.shape

        I = dict()

        # Encode the number of clusters, K.
        #   I(K) = K\log{2} + constant # [nats]
        I["I_k"] = K * np.log(2) # [nats]

        # Encode the relative weights for all K clusters.
        #   I(w) = \frac{(K - 1)}{2}\log{N} - \frac{1}{2}\sum_{j=1}^{K}\log{w_j} - \log{(K - 1)!} # [nats]
        # Recall: \log{(K-1)!} = \log{|\Gamma(K)|}
        I["I_w"] = (K - 1) / 2.0 * np.log(N) \
                 - 0.5 * np.sum(np.log(self.weights)) \
                 - scipy.special.gammaln(K)

        # I_1 is as per page 299 of Wallace et al. (2005).

        #_factor_scores = np.dot(self._responsibility.T, self.cluster_factor_scores.T)


        residual = y - self.means - np.dot(self.approximate_factor_scores, self.factor_loads)

        S = np.sum(self.approximate_factor_scores)
        v_sq = np.sum(self.approximate_factor_scores**2)
        specific_sigmas = np.sqrt(self.specific_variances)
        
        b_sq = np.sum((self.factor_loads/specific_sigmas)**2)

        I["I_sigmas"] = (N - 1) * np.sum(np.log(specific_sigmas))
        I["I_a"] = 0.5 * (D * np.log(N * v_sq - S**2) + (N + D - 1) * np.log(1 + b_sq))
        if not np.isfinite(I["I_a"]):
            raise a
        I["I_b"] = 0.5 * v_sq 
        I["I_c"] = 0.5 * np.sum(residual**2/self.specific_variances)

        #I["lattice"] = 0.5 * N_free_params * log_kappa(N_free_params) 
        #I["constant"] = 0.5 * N_free_params

        I_total = np.hstack(I.values()).sum() / np.log(2) # [bits]

        print(I)
        print(np.median(residual), self.specific_variances**0.5)

        if not full_output:
            return I_total

        for key in I.keys():
            I[key] /= np.log(2) # [bits]

        return (I_total, I)


def _iterate_wallace_freeman_1990(beta, N, D, V):

    beta = beta.reshape((D, ))

    b_squared = np.sum(beta**2)

    lhs = (N - 1.0) * b_squared
    if lhs <= D:
        logger.warn("Setting \Beta_k = 0 as {} <= {}".format(lhs, D))
        beta = np.zeros(D)

    # Compute specific variances according to:
    #   \sigma_k^2 = \frac{\sum_{n}w_{nk}^2}{(N-1)(1 + \Beta_k^2)}
    # (If \Beta = 0, exit)
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