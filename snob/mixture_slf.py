
"""
An estimator to model data using a prescribed number of multivariate Gaussian 
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
    Model data from (potentially) many multivariate Gaussian distributions
    with a single (multivariate) latent factor.

    The latent factor and mixture parameters are iteratively updated using
    expectation-maximization, with minimum message length describing the cost
    function.

    :param num_components:
        The number of multivariate Gaussian mixtures to model.

    :param tolerance: [optional]
        The relative improvement in log probability required before stopping
        an expectation-maximization step (default: ``1e-5``).

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a full covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``full``).

    :param covariance_regularization: [optional]
        Regularization strength to apply to the diagonal of covariance
        matrices (default: ``0``)

    :param max_iter: [optional]
        The maximum number of iterations to run per expectation-maximization
        loop (default: ``10000``).

    :param initialization_method: [optional]
        The method to use to initialize the mixture parameters. Available
        options are: ``kmeans``, and ``random`` (default: ``kmeans``).
    """

    parameter_names = (
        "factor_loads", "factor_scores", "specific_variances",
        "means", "covariances", "weights")

    def __init__(self, num_components, tolerance=1e-5, covariance_type="full", 
        covariance_regularization=0, max_iter=10000, 
        initialization_method="kmeans", **kwargs):

        num_components = int(num_components)
        if 1 > num_components:
            raise ValueError("number of components must be a positive integer")

        if 0 >= tolerance:
            raise ValueError("tolerance must be a positive value")

        available = ("full", "diag", "tied", "tied_diag")
        covariance_type = covariance_type.strip().lower()
        if covariance_type not in available:
            raise ValueError(
                "Covariance type '{}' is invalid. Must be one of: {}"\
                .format(covariance_type, ", ".join(available)))

        assert covariance_type == "full"

        if 0 > covariance_regularization:
            raise ValueError("covariance_regularization must be non-negative")
        
        if 0 >= max_iter:
            raise ValueError("max_iter must be a positive integer")

        initialization_method = initialization_method.strip().lower()
        available = ("kmeans", "random")
        if initialization_method not in available:
            raise ValueError(
                "Initialization method '{}' is invalid. Must be one of: {}"\
                .format(initialization_method, ", ".join(available)))

        self.num_components = num_components
        self.tolerance = tolerance
        self.covariance_type = covariance_type
        self.covariance_regularization = covariance_regularization
        self.max_iter = max_iter
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


    def initialize_parameters(self, y, **kwargs):
        r"""
        Initialize the model parameters, given the data.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        # Estimate the latent factor first.
        N, K = y.shape

        means = np.mean(y, axis=0)
        w = y - means
        correlation_matrix = np.corrcoef(w.T)

        U, S, V = np.linalg.svd(correlation_matrix)

        # Set the initial values for b as the principal eigenvector, scaled by
        # the square-root of the principal eigenvalue
        b = V[0] * S[0]**0.5

        specific_sigmas = np.sqrt(np.diag(np.dot(w.T, w)) \
                    / ((b*b + 1.0) * (N - 1)))
        factor_loads = specific_sigmas * b
        _y = (y - means)/specific_sigmas

        b_sq = np.sum(b**2)
        factor_scores = np.dot(_y, b) * (1 - K/(N - 1) * b_sq)/(1 + b_sq)
        specific_variances = specific_sigmas**2

        # Subtract off the factor loads and factor scores, then estimate the
        # cluster properties.
        factor_scores = np.atleast_2d(factor_scores)
        factor_loads = np.atleast_2d(factor_loads)
        y_nlf = y - np.dot(factor_scores.T, factor_loads)

        if self.initialization_method == "kmeans":

            # Do k-means on the data without the latent factor.
            responsibility = np.zeros((N, self.num_components))
            kmeans = cluster.KMeans(
                n_clusters=self.num_components, n_init=1, **kwargs)
            labels = kmeans.fit(y_nlf).labels_
            responsibility[np.arange(N), labels] = 1

        elif self.initialization_method == "random":
            responsibility = np.random.randint(0, self.num_components, size=N)
            responsibility /= responsibility.sum(axis=1)[:, np.newaxis]

        else:
            raise ValueError("Unknown initialization method: {}".format(
                self.initialization_method))

        return self._maximization(y, responsibility)
        """
        # Get the means
        self.set_para
        
        raise a

        N, K = y.shape





        means = np.atleast_2d(np.mean(y, axis=0))

        # Initialize the latent factors first, and then the mixture parameters
        U, S, V = np.linalg.svd(np.cov(y.T))

        # Set the largest eigenvector as the initial latent factor.
        factor_loads = V[0] # a_k
        factor_scores = np.ones(N) # v_n
        specific_variances = np.ones(K) # \sigma_k


        responsibility = np.ones((N, 1))

        self.set_parameters(means, factor_scores, factor_loads, specific_variances)

        return self._maximization(y, responsibility)
        """

    def _soft_initialize(self, y, **kwargs):
        """
        Initialize the mixture parameters, only if no mixture parameters exist.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :returns:
            ``True`` or ``False`` whether the mixture parameters were updated.
        """

        for parameter_name in self.parameter_names:
            try:
                value = getattr(self, parameter_name)
            except AttributeError:
                break

            else:
                if value is None:
                    break

        else:
            # All parameters exist and are not None
            return False

        # Set the parameters.
        return self.initialize_parameters(y, **kwargs)


    def set_parameters(self, *args):
        r"""
        Set the Gaussian mixture parameters and their relative weights.

        
        :returns:
            ``True`` or ``False`` if the parameters were successfully set.

        :raise ValueError:
            If there is a shape mis-match between the input arrays.
        """

        for parameter_name, arg in zip(self.parameter_names, args):
            arg = arg if arg is None else np.atleast_2d(arg)
            setattr(self, "_{}".format(parameter_name), arg)


        return False


    def fit(self, x, **kwargs):
        r"""
        Fit the mixture model to the data using the expectation-maximization
        algorithm, and minimum message length to update the parameter
        estimates.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        foo = self.initialize_parameters(x)

        raise a

        # Solve for latent factors first.
        N, K = x.shape

        means = np.mean(x, axis=0)

        w = x - means
        correlation_matrix = np.corrcoef(w.T)

        U, S, V = np.linalg.svd(correlation_matrix)

        # Set the initial values for b as the principal eigenvector, scaled by
        # the square-root of the principal eigenvalue
        b = V[0] * S[0]**0.5

        for iteration in range(self.max_iter):

            # The iteration scheme is from Wallace & Freeman (1992)
            if (N - 1) * np.sum(b**2) <= K:
                b = np.zeros(K)

            specific_variances = np.sum(w**2, axis=0) \
                               / ((N - 1) * (1 + b**2))
            specific_sigmas = np.sqrt(np.atleast_2d(specific_variances))

            b_sq = np.sum(b**2)
            if b_sq == 0:
                raise NotImplementedError("a latent factor is not justified")
                break

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

            if self.tolerance >= change:
                break

        specific_sigmas = np.sqrt(np.diag(np.dot(w.T, w)) \
                    / ((b*b + 1.0) * (N - 1)))
        factor_loads = specific_sigmas * b
        y = (x - means)/specific_sigmas

        b_sq = np.sum(b**2)
        factor_scores = np.dot(y, b) * (1 - K/(N - 1) * b_sq)/(1 + b_sq)
        specific_variances = specific_sigmas**2
        
        means = means.reshape((1, -1))
        factor_loads = factor_loads.reshape((1, -1))
        factor_scores = factor_scores.reshape((-1, 1))
        specific_variances = specific_variances.reshape((1, -1))

        # Set the parameters.
        self.set_parameters(means, factor_loads, factor_scores, 
            specific_variances)
        self.meta.update(iterations=iteration, delta_sum_abs_b=change)

        #L1 = np.sum((x - means - np.dot(factor_scores, np.atleast_2d(factor_loads).T))**2/specific_variances)

        return self


    def _expectation(self, y):
        r"""
        Perform the expectation step of the expectation-maximization algorithm.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :returns:
            The log of the normalized probabilities for all :math:`N` objects,
            and the log of the responsibility matrix.
        """

        weighted_log_prob = self._estimate_weighted_log_prob(y)
        #log_prob_norm = scipy.misc.logsumexp(weighted_log_prob, axis=1)
        #with np.errstate(under="ignore"):
        #    # Ignore underflow errors.
        #    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        log_prob_norm = weighted_log_prob

        N, K = y.shape
        return (log_prob_norm, np.zeros((N, 1)))


    def _maximization(self, y, responsibility):
        r"""
        Perform the maximization step of the expectation-maximization algorithm
        on all components.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :param responsibility: 
            The responsibility matrix for all :math:`N` observations being
            partially assigned to each :math:`K` component.
        """


        # Update the estimate of the latent factor, after subtracting the
        # means from each cluster.
        N, M = responsibility.shape
        N, K = y.shape

        effective_membership = np.sum(responsibility, axis=0)

        means = np.zeros((M, K))
        for m in range(M):
            means[m] = np.sum(responsibility[:, m] * y.T, axis=1) \
                     / effective_membership[m]

        # Subtract the means, weighted by responsibilities.
        w_means = np.dot(responsibility, means)

        w = y - w_means
        correlation_matrix = np.corrcoef(w.T)

        U, S, V = np.linalg.svd(correlation_matrix)

        # Set the initial values for b as the principal eigenvector, scaled by
        # the square-root of the principal eigenvalue
        b = V[0] * S[0]**0.5

        # Iteratively solve for the vector b.
        for iteration in range(self.max_iter):

            # The iteration scheme is from Wallace & Freeman (1992)
            if (N - 1) * np.sum(b**2) <= K:
                b = np.zeros(K)

            specific_variances = np.sum(w**2, axis=0) \
                               / ((N - 1) * (1 + b**2))
            specific_sigmas = np.sqrt(np.atleast_2d(specific_variances))

            b_sq = np.sum(b**2)
            if b_sq == 0:
                raise NotImplementedError("a latent factor is not justified")
                break

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
            if self.tolerance >= change:
                break

        specific_sigmas = np.sqrt(np.diag(np.dot(w.T, w)) \
                    / ((b*b + 1.0) * (N - 1)))
        factor_loads = specific_sigmas * b
        scaled_y = (y - w_means)/specific_sigmas

        b_sq = np.sum(b**2)
        factor_scores = np.dot(scaled_y, b) * (1 - K/(N - 1) * b_sq)/(1 + b_sq)
        specific_variances = specific_sigmas**2
        
        factor_loads = factor_loads.reshape((1, -1))
        factor_scores = factor_scores.reshape((-1, 1))
        specific_variances = specific_variances.reshape((1, -1))

        # Update the covariance matrices of the clusters.
        # TODO: this should be done differently, because the specific variances
        covariances = np.zeros((M, K, K))
        for m in range(M):

            diff = y - w_means - np.dot(factor_scores, factor_loads)
            denom = effective_membership[m]
            denom = denom - 1 if denom > 1 else denom

            covariances[m] = np.dot(responsibility[:, m] * diff.T, diff) \
                           / denom
            covariances[m][::M + 1] += self.covariance_regularization

        weights = (effective_membership + 0.5)/(N + M/2.0)

        return self.set_parameters(
            factor_loads, factor_scores, specific_variances,
            means, covariances, weights)



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


    def _estimate_weighted_log_prob(self, y):
        r"""
        Estimate the weighted log probability of the observations :math:`y`
        belonging to each Gaussian mixture.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """
        return self._estimate_log_prob(y) + self._estimate_log_weights()


    def _estimate_log_weights(self):
        r"""
        Return the natural logarithm of the mixture weights.
        """
        return 0.0

        #return np.log(self.weights)


    def _estimate_log_prob(self, y):
        r"""
        Return the estimated log probabilities of the observations :math:`y`
        belonging to each Gaussian mixture.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """
        return _estimate_log_latent_factor_prob(
            y, self.means, self.factor_scores, self.factor_loads,
            self.specific_variances)

def _compute_log_det_cholesky(matrix_chol, covariance_type, D):
    r"""
    Compute the log-determinate of the Cholesky decomposition of matrices.


    :param matrix_chol:
        An array-like object containing the Cholesky decomposition of the
        matrices. Expected shape is 
            ``(M, D, D)`` for ``covariance_type='full'``,
            ``(D, D)`` for ``covariance_type='tied'``, 
            ``(M, D)`` for ``covariance_type='diag'``, and
            ``(M, )`` for ``covariance_type='spherical'``, 
        where ``M`` is the number of mixtures and ``D`` is the number of
        dimensions.

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a full covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components.
 
    :param D:
        The number of dimensions, :math:`D`.

    :returns:
        The determinant of the precision matrix for each component.
    """

    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def _estimate_log_gaussian_prob(y, means, precision_cholesky, covariance_type):
    r"""
    Compute the logarithm of the probabilities of the observations belonging
    to the Gaussian mixtures.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is
        the number of dimensions per observation.

    :param means:
        The estimates of the mean values of the Gaussian mixtures.

    :param precision_cholesky:
        The precision matrices of the Cholesky decompositions of the covariance
        matrices of the Gaussian mixtures.

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a full covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components.

    :returns:
        The log probabilities of the observations belonging to the Gaussian
        mixtures.
    """

    N, D = y.shape
    M, D = means.shape

    # det(precision_chol) is -half of det(precision)
    log_det = _compute_log_det_cholesky(precision_cholesky, covariance_type, D)

    if covariance_type == 'full':
        log_prob = np.empty((N, M))
        for k, (mu, prec_chol) in enumerate(zip(means, precision_cholesky)):
            diff = np.dot(y, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(diff), axis=1)

    return -0.5 * (D * np.log(2 * np.pi) + log_prob) + log_det


def _estimate_log_latent_factor_prob(y, means, factor_scores, factor_loads,
    specific_variances):

    N, K = y.shape
    M, K = means.shape

    specific_inverse_variances = 1.0/specific_variances
    log_prob = np.empty((N, M))
    for m, mu in enumerate(means):
        squared_diff = (y - mu - np.dot(factor_scores, factor_loads))**2

        raise a

    squared_diff = (y - means - np.dot(factor_scores, factor_loads))**2

    return - 0.5 * K * N * np.log(2 * np.pi) \
           + N * np.sum(np.log(specific_variances)) \
           + 0.5 * np.sum(squared_diff * specific_inverse_variances)


def _compute_cholesky_decomposition(covariances, covariance_type):
    r"""
    Compute the Cholesky decomposition of the given covariance matrices.

    :param covariances:
        An array of covariance matrices.

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a full covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components.

    :returns:
        The Cholesky decomposition of the given covariance matrices.
    """

    if covariance_type in "full":
        M, D, _ = covariances.shape

        cholesky_decomposition = np.empty((M, D, D))
        for m, covariance in enumerate(covariances):
            try:
                cholesky_cov = scipy.linalg.cholesky(covariance, lower=True) 
            except scipy.linalg.LinAlgError:
                raise ValueError(singular_matrix_error)

            cholesky_decomposition[m] = scipy.linalg.solve_triangular(
                cholesky_cov, np.eye(D), lower=True).T

    else:
        raise NotImplementedError("nope")

    return cholesky_decomposition


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