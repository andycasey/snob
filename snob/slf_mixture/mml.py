
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
        
        w = data - means
        V = np.dot(w.T, w)
        
        # Set \beta as the dominant eigenvector of the correlation matrix with
        # length given by setting b^2 equal to the dominant eigenvalue.
        _, s, v = np.linalg.svd(np.corrcoef(data.T))
        beta = v[0] * np.sqrt(s[0])

        # Iterate as per Wallace & Freeman first.
        for iteration in range(self.max_em_iterations):

            b_squared = np.sum(beta**2)

            lhs = (N - 1.0) * b_squared
            if lhs <= D:
                logger.warn("Setting \Beta = 0 as {} <= {}".format(lhs, D))
                beta = np.zeros(D)

            # Compute specific variances according to:
            #   \sigma_k^2 = \frac{\sum_{n}w_{nk}^2}{(N-1)(1 + \Beta_k^2)}
            # (If \Beta = 0, exit)
            specific_variances = np.diag(V) / ((N - 1.0) * (1.0 + beta**2))

            if np.all(beta == 0):
                logger.warn("Exiting step (b) of scheme in initialisation")
                break

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
            sum_l2 = np.sum((beta - beta_new)**2)
            print(iteration, sum_l2, beta, beta_new, )

            if not np.isfinite(sum_l2):
                raise wtf

            # HACK TODO MAGIC
            if sum_l2 < 1e-16:
                break

            beta = beta_new


        factor_loads = specific_sigmas * beta

        b_sq = np.sum(beta**2)
        factor_scores = np.dot(w/specific_sigmas, beta) \
                      * ((1.0 - D/((N - 1.0) * b_sq)) / (1 + b_sq))
        factor_scores = np.atleast_2d(factor_scores).T

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
                              / (effective_membership - 1)

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
        
        K, N = responsibility.shape
        N, D = data.shape

        # Recall:
        #   N is the number of things
        #   D is the number of dimensions per thing
        #   K is the number of clusters

        # Order of updates:
        # (1) Update the cluster factor scores 

        # In Wallace & Freeman (1990) the MML estimate is given by Eq (4.4),
        # where in their nomenclature:
        #   v_{n} = y_{n}\dot\Beta/(1 + b^2 + K/v^2)
        # where in our nomenclature:
        #   v_{n} = y_{n}\dot\Beta/(1 + b^2 + D/v^2)

        beta = self.factor_loads/np.sqrt(self.specific_variances)
        effective_membership = responsibility.sum(axis=1)

        b_squared = np.sum(beta**2)
        v_squared = np.sum(effective_membership * self.cluster_factor_scores**2)

        factor_scores = np.dot(data, beta.T)/(1.0 + b_squared + float(D)/v_squared)
        cluster_factor_scores = np.dot(responsibility, factor_scores).T \
                              / (effective_membership - 1.0)

        cluster_factor_scores = np.atleast_2d(cluster_factor_scores)
        self.set_parameters(cluster_factor_scores=cluster_factor_scores)

        # (2) Update the expectation
        #responsibility, _ = self._expectation(data)
        #self._aecm_step_1(data, responsibility)

        # (3) Update the factor loads
        
        # In Wallace & Freeman (1990) the MML estimate for \Beta_k is given by
        #   \Beta_{k} = \sum_{n} y_{nk}v_{n}/(N - 1)
        # which is the same in our nomenclature and theirs (finally!)

        # Note: could use self.approximate_factor_scores here but I am doing
        #       this here so that it is clear this makes use of the updated
        #       responsibility matrix.
        approximate_factor_scores = np.dot(
            responsibility.T, cluster_factor_scores.T)

        beta = np.sum(data * approximate_factor_scores, axis=0)/(N - 1.0)
        factor_loads = beta * np.sqrt(self.specific_variances)
        self.set_parameters(factor_loads=factor_loads)

        
        # (4) Update the expectation
        #responsibility, _ = self._expectation(data)
        #self._aecm_step_1(data, responsibility)

        # (5) Update the specific sigmas
        # In Wallace & Freeman (1990) the MML estimate for the specific
        # variances can be found from eq (4.9):
        #   (N - 1)\sigma_{k}^{2} = \frac{\sum_{n}w_{nk}^2}{(N - 1)(1 + \Beta_{k}^2}
        # or similarly from eq (4.10):
        #   (N - 1)\sigma_{k}^{2} = \sum_{n} (w_{nk} - v_{n}a_{k})^2 + a_{k}^{2}(N - 1 - v^{2})
        # Which is the same in our nomenclature or ours.

        # Eq 4.10:
        #effective_membership = responsibility.sum(axis=1)
        #v_squared = np.sum(effective_membership * cluster_factor_scores**2)
        #rhs = (w - np.dot(factor_scores, factor_loads))**2 + factor_loads**2 * (N - 1.0 - v_squared)
        #specific_variances = np.sum(rhs, axis=0)/(N - 1.0)

        beta = factor_loads/np.sqrt(self.specific_variances)
        w = data - self.means
        rhs = np.sum(w**2, axis=0) / ((N - 1.0) * (1.0 + beta**2))
        specific_variances = rhs / (N - 1.0)

        self.set_parameters(specific_variances=specific_variances)

        return True

        raise a




        I = [self.message_length(data)]

        # NOTE THE TERMINOLOGY DIFFERENCE
        # \Beta_k refers to a_k/\sigma_k 
        # b^2 refers to \sum_{k}\Beta_k^2

        beta = self.factor_loads/specific_sigmas 






        raise a

        beta = np.sum(data * self.approximate_factor_scores, axis=0) \
             / (N - 1.0)

        for iteration in range(self.max_em_iterations):

            # In Wallace & Freeman nomenclature:
            #   If (N - 1) * b^2 <= K, set \Beta = 0
            # Which is, in our nomenclature:
            #   If (N - 1) * b^2 <= D, set \Beta = 0
            lhs = (N - 1.0) * np.sum(beta**2)
            if lhs <= D:
                logger.warn("Setting \Beta = 0 because {} <= {}".format(lhs, D))
                beta = np.zeros(D, dtype=float)

            #else:
            #    beta = np.sum(data * self.approximate_factor_scores, axis=0) \
            #         / (N - 1.0)

            # Compute specific variances according to:
            #   \sigma_k^2 = \frac{\sum_{n}w_{nk}^2}{(N-1)(1 + \Beta_k^2)}
            # (If \Beta = 0, exit)
            specific_variances = np.sum(w**2, axis=0) \
                               / ((N - 1.0) * (1.0 + beta**2))

            if np.all(beta == 0):
                logger.warn("Exiting step (b) of scheme in _aecm_step_2")
                break

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
            beta_new = (np.dot(Y, beta.flatten()) * (1.0 - D/(N - 1.0) * b_squared)) \
                     / ((N - 1.0) * (1.0 + b_squared))

            # If \Beta_new \approx \Beta then exit.
            abs_difference = np.sum(np.abs(beta - beta_new))
            print(iteration, abs_difference, beta, beta_new, )

            # TODO MAGIC HACK
            
            if not np.isfinite(abs_difference):
                raise wtf

            if abs_difference < 1e-8:
                break



            beta = beta_new


        factor_loads = specific_sigmas * beta

        b_sq = np.sum(beta**2)
        factor_scores = np.dot(data, beta.flatten()) * (1.0 - D/(N - 1.0) * b_sq) \
                      / (1.0 + b_sq)

        # Calculate cluster factor scores.
        effective_membership = np.sum(responsibility, axis=1)
        cluster_factor_scores = np.dot(responsibility, factor_scores).T \
                              / (effective_membership - 1.0)

        cluster_factor_scores = np.atleast_2d(cluster_factor_scores)
        specific_variances = np.atleast_2d(specific_variances)

        return self.set_parameters(
            factor_loads=factor_loads,
            cluster_factor_scores=cluster_factor_scores,
            specific_variances=specific_variances)









        # All below is wrong due to nomenclature!



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
        N, D = data.shape

        # Subtract the means, weighted by responsibilities.
        #w_means = np.dot(responsibility.T, self.means)
        w = data - self.means
        
        b = (self.factor_loads/np.sqrt(self.specific_variances)).flatten()

        I = [self.message_length(data)]
        for iteration in range(self.max_em_iterations):

            I.append(self.message_length(data))
            logger.debug("_aecm_step_2: iteration {} {}".format(iteration,
                I[-1]))


            # The iteration scheme is from Wallace & Freeman (1992)
            if (N - 1) * np.sum(b**2) <= D:
                b = np.zeros(D)

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

            new_b = (np.dot(Y, b) * (1 - D/(N - 1) * b_sq)) \
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

            b_sq = np.sum(b**2)
            specific_sigmas = np.sqrt(np.diag(np.dot(w.T, w)) / ((b_sq + 1.0) * (N - 1)))
            factor_loads = specific_sigmas * b
            scaled_y = w/specific_sigmas

            print("specific sigmas {}".format(specific_sigmas))
            raise a

            factor_scores = np.dot(scaled_y, b) * (1 - D/(N - 1) * b_sq)/(1 + b_sq)
            specific_variances = specific_sigmas**2

            factor_loads = factor_loads.reshape((1, -1))
            factor_scores = factor_scores.reshape((-1, 1))
            specific_variances = specific_variances.reshape((1, -1))

            effective_membership = responsibility.sum(axis=1)
            
            cluster_factor_scores = np.dot(responsibility, factor_scores).T \
                                  / (effective_membership - 1)

            self.set_parameters(
                factor_loads=factor_loads,
                cluster_factor_scores=cluster_factor_scores,
                specific_variances=specific_variances)

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
        
        cluster_factor_scores = np.dot(responsibility, factor_scores).T \
                              / (effective_membership - 1)

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
        _factor_scores = np.dot(self._responsibility.T, self.cluster_factor_scores.T)


        residual = y - self.means - np.dot(_factor_scores, self.factor_loads)

        S = np.sum(_factor_scores)
        v_sq = np.sum(_factor_scores**2)
        specific_sigmas = np.sqrt(self.specific_variances)
        
        b_sq = np.sum((self.factor_loads/specific_sigmas)**2)

        I["I_sigmas"] = (N - 1) * np.sum(np.log(specific_sigmas))
        I["I_a"] = 0.5 * (D * np.log(N * v_sq - S**2) + (N + D - 1) * np.log(1 + b_sq))
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