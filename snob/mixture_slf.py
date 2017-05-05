
"""
An estimator to model data using a fixed number of multivariate Gaussian 
distributions with a single multivariate latent factor.
"""

__all__ = ["SLFGaussianMixture"]

import logging
import numpy as np
import scipy
from sklearn import cluster

logger = logging.getLogger(__name__)


class SLFGaussianMixture(object):

    r"""
    Model data from a fixed number of multivariate Gaussian distributions with
    a single (multivariate) latent factor.

    The latent factor and mixture parameters are iteratively updated using
    expectation-maximization, with minimum message length describing the cost
    function.

    :param num_mixtures:
        The number of multivariate Gaussian mixtures to model.

    :param threshold: [optional]
        The relative improvement in log probability required before stopping
        an expectation-maximization step (default: ``1e-5``).

    :param max_em_iterations: [optional]
        The maximum number of iterations to run per expectation-maximization
        loop (default: ``10000``).

    :param max_inner_iterations: [optional]
        The maximum number of iterations to run when updating the latent
        factor in the maximization step (default: ``1000``).

    :param initialization_method: [optional]
        The method to use to initialize the mixture parameters. Available
        options are: ``kmeans``, and ``random`` (default: ``kmeans``).
    """

    parameter_names = ("factor_loads", "factor_scores", "specific_variances",
        "means", "weights")

    def __init__(self, num_mixtures, threshold=1e-5, max_em_iterations=10000,
        max_inner_iterations=1000, initialization_method="kmeans", **kwargs):

        num_mixtures = int(num_mixtures)
        if 1 > num_mixtures:
            raise ValueError("number of mixtures must be a positive integer")

        if 0 >= threshold:
            raise ValueError("threshold must be a positive value")

        if 0 >= max_em_iterations:
            raise ValueError("max_em_iterations must be a positive integer")

        available = ("kmeans", "random")
        initialization_method = initialization_method.strip().lower()
        if initialization_method not in available:
            raise ValueError(
                "Initialization method '{}' is invalid. Must be one of: {}"\
                .format(initialization_method, ", ".join(available)))

        self.num_mixtures = num_mixtures
        self.threshold = threshold
        self.max_em_iterations = max_em_iterations
        self.max_inner_iterations = max_inner_iterations
        self.initialization_method = initialization_method
        self.meta = {}

        self.set_parameters(**dict([(p, None) for p in self.parameter_names]))

        return None


    @property
    def means(self):
        r"""
        Return the means, :math:`\mu_k`.
        """
        return self._means


    @property
    def weights(self):
        r"""
        Return the mixture weights, :math:`\w_k`.
        """
        return self._weights


    @property
    def factor_loads(self):
        r"""
        Return the factor loads, :math:`a_k`.
        """
        return self._factor_loads


    @property
    def factor_scores(self):
        r"""
        Return the factor scores, :math:`v_n`.
        """
        return self._factor_scores


    @property
    def specific_variances(self):
        r"""
        Return the specific variances, :math:`\sigma_k^2`.
        """
        return self._specific_variances


    def initialize_parameters(self, data, **kwargs):
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

        b_sq = np.sum(b**2)
        factor_scores = np.dot(y, b) * (1 - K/(N - 1) * b_sq)/(1 + b_sq)
        specific_variances = specific_sigmas**2
        specific_variances = specific_variances.reshape((1, -1))

        # Subtract off the factor loads and factor scores, then estimate the
        # cluster properties.
        factor_scores = np.atleast_2d(factor_scores).reshape((-1, 1))
        factor_loads = np.atleast_2d(factor_loads).reshape((1, -1))

        y_nlf = data - np.dot(factor_scores, factor_loads)

        if self.initialization_method == "kmeans":

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


    def set_parameters(self, **kwargs):
        r"""
        Set specific parameters.
        """

        for parameter_name, value in kwargs.items():
            if parameter_name in self.parameter_names:
                value = value if value is None else np.atleast_2d(value)
                setattr(self, "_{}".format(parameter_name), value)
            else:
                raise ValueError("unknown parameter '{}'".format(parameter_name))

        return True


    def fit(self, data, **kwargs):
        r"""
        Fit the mixture model to the data using the expectation-maximization
        algorithm, and minimum message length to update the parameter
        estimates.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        # parameters contains:
        # (factor_loads, factor_scores, specific_variances, means, weights)
        initial_parameters = self.initialize_parameters(data)

        N, D = data.shape
        iterations = 1

        # Calculate the inital expectation.
        responsibility, log_likelihood = self._expectation(data) 
        results = [log_likelihood.sum()]

        for iteration in range(self.max_em_iterations):

            # Do conditional updates.
            # AECM
            # Do the CM-step 1, where we update the weights and means.
            self._conditional_maximization_1(data, responsibility)

            # Compute the e-step of cycle 2
            responsibility, _ = self._expectation(data)

            # Do the M-step of cycle 2
            self._conditional_maximization_2(data, responsibility)

            # Re-compute the expectation,
            responsibility, log_likelihood = self._expectation(data)
            results.append(log_likelihood.sum())

            # Check for convergence.
            change = np.abs((results[-1] - results[-2])/results[-2])
            
            print("E-M step {}: {} {}".format(
                iteration, results[-1], change))
            
            if change <= self.threshold \
            or iteration >= self.max_em_iterations:
                break

        return self


    def _expectation(self, data):
        r"""
        Perform the expectation step of the expectation-maximization algorithm.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :returns:
            The log of the normalized probabilities for all :math:`N` objects,
            and the log of the responsibility matrix.
        """

        weighted_log_prob = self._estimate_weighted_log_prob(data)

        log_likelihood = scipy.misc.logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # Ignore underflow errors.
            log_resp = weighted_log_prob - log_likelihood[:, np.newaxis]

        responsibility = np.exp(log_resp).T
        
        return (responsibility, log_likelihood)


    def _conditional_maximization_1(self, data, responsibility):
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

        effective_membership = np.sum(responsibility, axis=1)
        denominator = effective_membership.copy()
        denominator[denominator > 1] = denominator[denominator > 1] - 1

        means = np.zeros((M, K))
        for m in range(M):
            means[m] = np.sum(responsibility[m] * data.T, axis=1) \
                     / denominator[m]

        weights = (effective_membership + 0.5)/(N + M/2.0)

        return self.set_parameters(weights=weights, means=means)


    def _ecm_conditional_maximization_2(self, data, responsibility):
        r"""
        Perform the second conditional maximization step of the AECM algorithm,
        where we update the factor loads, given the current specific variances
        and the (recently updated) weights and means.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :param responsibility: 
            The responsibility matrix for all :math:`N` observations being
            partially assigned to each :math:`K` component.
        """

        M, N = responsibility.shape
        N, K = data.shape

        #A(t+1) = Uj(\Lambda_j - I)**0.5 
        #Uj = means
        # Lambda_j = diag(\lambda_1, \lambda_2, etc)
        # where \lambda_1 is the first eigenvector of S.
        denominator = N * self.weights.flatten()
        denominator[denominator > 1] = denominator - 1

        M = self.weights.size
        S_tilde = np.zeros((M, N))
        for m in range(M):
            residual = data - self.means[m]
            S = np.sum(responsibility[m] * np.dot(residual, residual.T), axis=0) / denominator[m]

            #S = np.sum(residual**2, axis=0)/denominator[m]
            _ = (self.specific_variances[0]**(-0.5)).reshape((1, -1))

            S_tilde[m] = np.dot(_, np.dot(S.reshape((-1, 1)), _).T)[0]
            raise a

        #   S_tilde = 

        U, S, V = np.linalg.svd(S_tilde)

        raise a

        

    def _conditional_maximization_2(self, data, responsibility):
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
        
        M, N = responsibility.shape
        N, K = data.shape

        # Subtract the means, weighted by responsibilities.
        w_means = np.dot(responsibility.T, self.means)
        w = data - w_means
        
        # Get best current estimate of b
        assert self.factor_loads is not None
        assert self.specific_variances is not None

        b = (self.factor_loads/np.sqrt(self.specific_variances)).flatten()

        for iteration in range(self.max_inner_iterations):

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
                print("Non-finite change on iteration {}".format(iteration))
                b = 0.5 * (self.factor_loads/np.sqrt(self.specific_variances)).flatten()
                break

            b = new_b        

            logger.debug(
                "Iteration #{} on SLF, delta: {}".format(iteration, change))

            #print(iteration, change)
            if self.threshold >= change:
                print("Tolerance achieved on iteration {}".format(iteration))
                break

        else:
            print("Scheme did not converge")
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


    def _estimate_weighted_log_prob(self, data):
        r"""
        Estimate the weighted log probability of the observations :math:`y`
        belonging to each Gaussian mixture.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """
        return self._estimate_log_prob(data) + self._estimate_log_weights()


    def _estimate_log_weights(self):
        r"""
        Return the natural logarithm of the mixture weights.
        """
        return np.log(self._weights)


    def _estimate_log_prob(self, data):
        r"""
        Return the estimated log probabilities of the observations :math:`y`
        belonging to each Gaussian mixture.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """
        return _estimate_log_latent_factor_prob(
            data, self.factor_loads, self.factor_scores, 
            self.specific_variances, self.means)


def _estimate_log_latent_factor_prob(data, factor_loads, factor_scores,
    specific_variances, means):

    N, K = data.shape
    M, K = means.shape

    specific_inverse_variances = 1.0/specific_variances
    log_prob = np.empty((N, M))
    for m, mu in enumerate(means):
        squared_diff = (data - mu - np.dot(factor_scores, factor_loads))**2
        log_prob[:, m] = np.sum(squared_diff * specific_variances, axis=1)

    return - 0.5 * K * N * np.log(2 * np.pi) \
           + N * np.sum(np.log(specific_variances)) \
           - 0.5 * log_prob


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