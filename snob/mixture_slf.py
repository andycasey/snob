
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

    parameter_names \
        = ("means", "factor_scores", "factor_loads", "specific_variances")

    def __init__(self, num_components, tolerance=1e-5, covariance_type="full", 
        covariance_regularization=0, max_iter=10000, 
        initialization_method="kmeans", **kwargs):

        assert num_components == 1

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


    def fit(self, x, foo1, foo2, foo3, **kwargs):
        r"""
        Fit the mixture model to the data using the expectation-maximization
        algorithm, and minimum message length to update the parameter
        estimates.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        N, K = x.shape

        # Solve for latent factors first.

        # Calculate means.
        means = np.mean(x, axis=0)

        w = x - means
        correlation_matrix = np.corrcoef(w.T)

        U, S, V = np.linalg.svd(correlation_matrix)

        """
        # Take b as the dominant eigenvector of the correlation matrix,
        # and set b^2 as the dominant eigenvalue.
        b = V[0] * np.sqrt(S[0])

        def message_length(b):

            b_squared = np.sum(b**2)

            specific_variances = np.sum(w**2, axis=0) \
                               / ((N - 1) * (1 + b))

            v_scale = (1 - K/((N - 1) * b_squared)) / (1 + b_squared)

            

            specific_sigma = np.sqrt(specific_variances)
            Y = np.dot(w.T, w)\
              / np.dot(specific_sigma.T, specific_sigma)


            Ybb = np.sum(np.dot(Y, np.dot(b.T, b)))
            wbv = Ybb * v_scale
            v2 = wbv * v_scale
            residual2 = (b_squared + 1.0) * (N - 1) - 2 * wbv + b_squared * v2

            foo= (N - 1) * np.sum(np.log(specific_sigma)) \
                +  0.5 * K * np.log(N * v2) \
                + 0.5 * (N - 1 + K) * np.log(1 + b_squared) \
                + 0.5 * v2 \
                + 0.5 * residual2
            print(b, foo)
            return foo


        import scipy.optimize as op

        b = op.fmin_l_bfgs_b(message_length, b, approx_grad=True)[0]

        #b = fmin(message_length, b, maxfun=10000000, maxiter=10000000)
        #import pickle
        #with open("b.pkl", "rb") as fp:
        #    b = pickle.load(fp)

        sigma = np.sqrt(np.diag(np.dot(w.T, w)) \
              / ((b * b + 1.0) * (N - 1)))

        factor_loads = sigma * b
        y = (x - means)/sigma

        b_squared = np.sum(b**2)
        factor_scores = y * b * (1 - K/(N - 1) * b_squared)/(1 + b_squared)

        specific_variances = sigma**2

        other_b = b.copy()
        """
        b = V[0] * S[0]**0.5

        for iteration in range(self.max_iter):

            if (N - 1) * np.sum(b**2) <= K:
                b = np.zeros(K)

            specific_variance = np.sum(w**2, axis=0) \
                               / ((N - 1) * (1 + b**2))
            specific_sigma = np.sqrt(np.atleast_2d(specific_variance))

            b_squared = np.sum(b**2)
            if b_squared == 0:
                break

            Y = np.dot(w.T, w) \
              / np.dot(specific_sigma.T, specific_sigma)

            new_b = (np.dot(Y, b) * (1 - K/(N - 1) * b_squared)) \
                  / ((N - 1) * (1 + b_squared))

            change = np.sum(np.abs(b - new_b))
            b = new_b

            print(iteration, change)

            if self.tolerance >= change:
                break

        
        specific_sigma = np.sqrt(np.diag(np.dot(w.T, w)) \
                    / ((b*b + 1.0) * (N - 1)))
        factor_loads = specific_sigma * b
        y = (x - means)/specific_sigma

        b_squared = np.sum(b**2)
        factor_scores = np.dot(y, b) * (1 - K/(N - 1) * b_squared)/(1 + b_squared)

        specific_variance = specific_sigma**2


        def message_length(b):

            b_squared = np.sum(b**2)

            sv2 = np.sum(w**2, axis=0) \
                               / ((N - 1) * (1 + b))

            v_scale = (1 - K/((N - 1) * b_squared)) / (1 + b_squared)

            

            ss = np.sqrt(sv2)
            Y = np.dot(w.T, w)\
              / np.dot(ss.T, ss)


            Ybb = np.sum(np.dot(Y, np.dot(b.T, b)))
            wbv = Ybb * v_scale
            v2 = wbv * v_scale
            residual2 = (b_squared + 1.0) * (N - 1) - 2 * wbv + b_squared * v2

            foo= (N - 1) * np.sum(np.log(ss)) \
                +  0.5 * K * np.log(N * v2) \
                + 0.5 * (N - 1 + K) * np.log(1 + b_squared) \
                + 0.5 * v2 \
                + 0.5 * residual2
            print(b, foo)
            return foo


        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3)

        axes[0].scatter(foo1[0], factor_loads, alpha=0.5)
        axes[0].set_xlabel("True factor load")
        axes[0].set_ylabel("Inferred factor load")

        limits = np.array([axes[0].get_xlim(), axes[0].get_ylim()])
        limits = (np.min(limits), np.max(limits))
        axes[0].set_xlim(limits)
        axes[0].set_ylim(limits)

        axes[1].scatter(foo2.T[0], factor_scores, alpha=0.5)
        axes[1].set_xlabel("True factor score")
        axes[1].set_ylabel("Inferred factor scores")

        limits = np.array([axes[1].get_xlim(), axes[1].get_ylim()])
        limits = (np.min(limits), np.max(limits))
        axes[1].set_xlim(limits)
        axes[1].set_ylim(limits)

        fig.tight_layout()
        fig.savefig("slf_experiment_v0.png", dpi=300)

        raise a



        # Only initialize if we don't have parameters already.
        self._soft_initialize(y, **kwargs)
        self.converged = False
        prev_log_prob_norm = -np.inf

        for iteration in range(self.max_iter):

            # Expectation.
            log_prob_norm, log_responsibility = self._expectation(y)

            # Maximization.
            self._maximization(y, np.exp(log_responsibility))

            # Check for convergence.
            mean_log_prob_norm = np.mean(log_prob_norm)

            change = prev_log_prob_norm - mean_log_prob_norm
            prev_log_prob_norm = mean_log_prob_norm

            print(iteration, prev_log_prob_norm, mean_log_prob_norm, change)
            print(self.means)

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
        #log_prob_norm = scipy.misc.logsumexp(weighted_log_prob, axis=1)
        #with np.errstate(under="ignore"):
        #    # Ignore underflow errors.
        #    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        log_prob_norm = weighted_log_prob

        N, K = y.shape
        return (log_prob_norm, np.zeros((N, 1)))


    def _maximization(self, x, responsibility, parent_responsibility=1):
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


        N, K = x.shape

        w = x - self.means
        y = w/self.specific_variances
        b = self.factor_loads/self.specific_variances
        b_squared = np.sum(b**2)
        #v_squared = np.sum(v**2)


        Y = np.dot(y.T, y)

        # New MML estimates.
        new_means = np.mean(x, axis=0)
        
        U, S, V = np.linalg.svd(Y/N)
        new_b = V[0]

        # TODO: use old b, or new b?
        new_specific_variances = np.sum(w**2, axis=0) \
                               / ((N - 1) * (1 + b_squared))

        new_factor_loads = new_b * np.sqrt(new_specific_variances)
        new_factor_scores = (1 - 1.0/N) * np.dot(y, b.T) / S[0]


        return self.set_parameters(new_means, new_factor_scores,
            new_factor_loads, new_specific_variances)



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

    specific_inverse_variances = 1.0/specific_variances
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