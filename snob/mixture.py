
"""
An estimator to model data using a prescribed number of multivariate Gaussian 
distributions with a prescribed number of multivariate latent factors.
"""

__all__ = ["GaussianMixture"]

import logging
import numpy as np
import scipy
from sklearn import cluster

logger = logging.getLogger(__name__)


class GaussianMixture(object):

    r"""
    Model data from (potentially) many multivariate Gaussian distributions, 
    using minimum message length.

    :param num_components:
        The number of multivariate Gaussian mixtures to model.

    :param num_latent_factors: [optional]
        The number of true latent factors (default: ``0``).

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

    parameter_names = ("means", "covariances", "weights")

    def __init__(self, num_components, num_latent_factors=0, tolerance=1e-5,
        covariance_type="full", covariance_regularization=0, max_iter=10000,
        initialization_method="kmeans", **kwargs):

        num_components = int(num_components)
        if 1 > num_components:
            raise ValueError("number of components must be a positive integer")

        num_latent_factors = int(num_latent_factors)
        if 0 > num_latent_factors:
            raise ValueError("number of latent factors must be non-negative")

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
        self.num_latent_factors = num_latent_factors
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
        Return the means of the Gaussian mixtures.
        """
        return self._means


    @property
    def covariances(self):
        r"""
        Return the covariance matrices of the Gaussian mixtures.
        """
        return self._covariances


    @property
    def weights(self):
        r"""
        Return the relative weights of the Gaussian mixtures.
        """
        return self._weights


    @property
    def precision_cholesky(self):
        r"""
        Return the Cholesky decompositions of the covariance matrices of the 
        Gaussian mixtures.
        """
        return self._precision_cholesky


    def initialize_parameters(self, y, **kwargs):
        r"""
        Initialize the mixture parameters, given the data.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        N, D = y.shape
        if self.initialization_method == "kmeans":
            responsibility = np.zeros((N, self.num_components))
            kmeans = cluster.KMeans(
                n_clusters=self.num_components, n_init=1, **kwargs)
            labels = kmeans.fit(y).labels_
            responsibility[np.arange(N), labels] = 1

        elif self.initialization_method == "random":
            responsibility = np.random.randint(0, self.num_components, size=N)
            responsibility /= responsibility.sum(axis=1)[:, np.newaxis]

        else:
            raise ValueError("Unknown initialization method: '{}'".format(
                self.initialization_method))

        return self._maximization(y, responsibility)


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


    def set_parameters(self, means, covariances, weights):
        r"""
        Set the Gaussian mixture parameters and their relative weights.

        :param means:
            A :math:`M\times{}D` array of the multivariate means for all
            :math:`M` Gaussian components.

        :param covariances:
            A :math:`M\times{}D\times{}D` array of the covariance matrices
            for all :math:`M` Gaussian components.

        :param weights:
            A :math:`M` array of the relative mixing weights for each
            Gaussian components. The relative weights must sum to one.
        
        :returns:
            ``True`` or ``False`` if the parameters were successfully set.

        :raise ValueError:
            If there is a shape mis-match between the input arrays.
        """

        if means is None or covariances is None or weights is None:
            for parameter_name in self.parameter_names:
                setattr(self, "_{}".format(parameter_name), None)
                return False

        means = np.atleast_2d(means)
        covariances = np.atleast_3d(covariances)
        weights = np.array(weights)

        M, D = means.shape
        if covariances.shape != (M, D, D):
            raise ValueError(
                "covariances have wrong expected shape ({M}, {D}, {D} != {a})"\
                .format(M=M, D=D, a=covariances.shape))

        if weights.size != M:
            raise ValueError(
                "weights have wrong expected size ({} != {})".format(
                    M, weights.size))

        self._means = means
        self._covariances = covariances
        self._weights = weights

        self._precision_cholesky = _compute_cholesky_decomposition(
            self.covariances, self.covariance_type)
        return True



    def fit(self, y, **kwargs):
        r"""
        Fit the mixture model to the data using the expectation-maximization
        algorithm, and minimum message length to update the parameter
        estimates.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        # Only initialize if we don't have parameters already.
        self._soft_initialize(y, **kwargs)
        self.converged = False

        prev_log_prob_norm = -np.inf

        for iteration in range(self.max_iter):

            # Expectation.
            log_prob_norm, log_responsibility = self._expectation(y)


            if iteration > 0:
                raise a
                
            # Maximization.
            self._maximization(y, np.exp(log_responsibility))

            # Check for convergence.
            mean_log_prob_norm = np.mean(log_prob_norm)

            change = prev_log_prob_norm - mean_log_prob_norm
            prev_log_prob_norm = mean_log_prob_norm

            if self.tolerance > abs(change):
                self.converged = True
                break



        else:
            logger.warn(
                "Hit maximum number of expectation-maximization iterations "
                "({})".format(self.max_iter))
        
        self.meta.update(
            delta_log_prob_norm=change, iterations=iteration,
            log_prob_norm=log_prob_norm)

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
        log_prob_norm = scipy.misc.logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # Ignore underflow errors.
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        return (log_prob_norm, log_resp)


    def _maximization(self, y, responsibility, parent_responsibility=1):
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

        :param parent_responsibility: [optional]
            An array of length :math:`N` giving the parent component 
            responsibilities (default: ``1``). Only useful if the maximization
            step is to be performed on sub-mixtures.
        """

        N, M = responsibility.shape
        N, D = y.shape

        effective_membership = np.sum(responsibility, axis=0)
        weights = (effective_membership + 0.5)/(N + M/2.0)

        w_responsibility = parent_responsibility * responsibility
        w_effective_membership = np.sum(w_responsibility, axis=0)

        means = np.zeros((M, D))
        covariances = np.zeros((M, D, D))

        for m in range(M):
            means[m] = np.sum(w_responsibility[:, m] * y.T, axis=1) \
                          / w_effective_membership[m]

            denom = w_effective_membership[m]
            denom = denom - 1 if denom > 1 else denom

            diff = y - means[m]
            covariances[m] = np.dot(w_responsibility[:, m] * diff.T, diff) \
                                / denom
            covariances[m][::M + 1] += self.covariance_regularization

        return self.set_parameters(means, covariances, weights)


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
        return np.log(self.weights)


    def _estimate_log_prob(self, y):
        r"""
        Return the estimated log probabilities of the observations :math:`y`
        belonging to each Gaussian mixture.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """
        return _estimate_log_gaussian_prob(
            y, self.means, self.precision_cholesky, self.covariance_type)


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
                n_components, -1)[:, ::D + 1]), 1))

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

    return  -0.5 * (D * np.log(2 * np.pi) + log_prob) + log_det
    

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