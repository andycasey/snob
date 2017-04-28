
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

        self.set_parameters(*([None] * len(self.parameter_names)))

        return None


    @property
    def means(self):
        r"""
        Return the means, :math:`\mu_k`.
        """
        return self._means


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

        # Subtract off the factor loads and factor scores, then estimate the
        # cluster properties.
        factor_scores = np.atleast_2d(factor_scores)
        factor_loads = np.atleast_2d(factor_loads)
        y_nlf = data - np.dot(factor_scores.T, factor_loads)

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

        # Return the estimate of the model parameters given the responsibility
        # matrix.
        return self._maximization(y, responsibility.T)


    def set_parameters(self, *args):
        r"""
        Set the model parameters.
        """

        if len(args) != len(self.parameter_names):
            raise ValueError("unexpected number of parameters ({} != {})"\
                .format(len(self.parameter_names), len(args)))

        for parameter_name, arg in zip(self.parameter_names, args):
            arg = arg if arg is None else np.atleast_2d(arg)
            setattr(self, "_{}".format(parameter_name), arg)
        return args


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

            parameters = self._maximization(data, responsibility)

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


    def _maximization(self, data, responsibility):
        r"""
        Perform the maximization step of the expectation-maximization algorithm
        on all mixtures.

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

        # Subtract the means, weighted by responsibilities.
        w_means = np.dot(responsibility.T, means)
        w = data - w_means
        
        # Get best current estimate of b
        if self.factor_loads is None:
            correlation_matrix = np.corrcoef(w.T)
            U, S, V = np.linalg.svd(correlation_matrix)
            b = V[0] * S[0]**0.5

        else:
            b = (self.factor_loads/np.sqrt(self.specific_variances)).flatten()

        for iteration in range(self.max_inner_iterations):

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
            assert np.isfinite(change)
            b = new_b        

            logger.debug(
                "Iteration #{} on SLF, delta: {}".format(iteration, change))

            print(iteration, change)
            if self.threshold >= change:
                break

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

        weights = (effective_membership + 0.5)/(N + M/2.0)

        return self.set_parameters(factor_loads, factor_scores, 
            specific_variances, means, weights)





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