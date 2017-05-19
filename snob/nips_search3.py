
"""
An estimator for modelling data from a mixture of Gaussians, 
using an objective function based on minimum message length.
"""

__all__ = [
    "GaussianMixture", 
    "kullback_leibler_for_multivariate_normals",
    "responsibility_matrix",
    "split_component", "merge_component", "delete_component", 
] 
import logging
import numpy as np
import scipy
from scipy import stats
import scipy.optimize as op
from collections import defaultdict
from sklearn.cluster import k_means_ as kmeans

logger = logging.getLogger(__name__)


class GaussianMixture(object):

    r"""
    Model data from (potentially) many multivariate Gaussian distributions, 
    using minimum message length (MML) as the objective function.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a free covariance matrix, or
        `diag` for a diagonal covariance matrix (default: ``diag``).

    :param covariance_regularization: [optional]
        Regularization strength to add to the diagonal of covariance matrices
        (default: ``0``).

    :param threshold: [optional]
        The relative improvement in message length required before stopping an
        expectation-maximization step (default: ``1e-5``).

    :param max_em_iterations: [optional]
        The maximum number of iterations to run per expectation-maximization
        loop (default: ``10000``).
    """

    parameter_names = ("mean", "covariance", "weight")

    def __init__(self, covariance_type="diag", covariance_regularization=1e-6, 
        threshold=1e-5, max_em_iterations=10000, predict_mixtures=10,
        **kwargs):

        available = ("full", "diag", )
        covariance_type = covariance_type.strip().lower()
        if covariance_type not in available:
            raise ValueError("covariance type '{}' is invalid. "\
                             "Must be one of: {}".format(
                                covariance_type, ", ".join(available)))

        if 0 > covariance_regularization:
            raise ValueError(
                "covariance_regularization must be a non-negative float")

        if 0 >= threshold:
            raise ValueError("threshold must be a positive value")

        if 1 > max_em_iterations:
            raise ValueError("max_em_iterations must be a positive integer")

        self._threshold = threshold
        self._max_em_iterations = max_em_iterations
        self._covariance_type = covariance_type
        self._covariance_regularization = covariance_regularization

        self._predict_mixtures = predict_mixtures
        self._mixture_predictors = []

        return None


    @property
    def covariance_type(self):
        r""" Return the type of covariance stucture assumed. """
        return self._covariance_type


    @property
    def covariance_regularization(self):
        r""" 
        Return the regularization applied to diagonals of covariance matrices.
        """
        return self._covariance_regularization


    @property
    def threshold(self):
        r""" Return the threshold improvement required in message length. """
        return self._threshold


    @property
    def max_em_iterations(self):
        r""" Return the maximum number of expectation-maximization steps. """
        return self._max_em_iterations


    def _fit_kasarapu_allison(self, y, **kwargs):
        r"""
        Minimize the message length of a mixture of Gaussians, 
        using the perturbation search algorithm described by
        Kasarapu & Allison (2015).

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.

        :returns:
            A tuple containing the optimized parameters ``(mu, cov, weight)``.
        """

        kwds = dict(
            threshold=self._threshold, 
            max_em_iterations=self._max_em_iterations,
            covariance_type=self.covariance_type, 
            covariance_regularization=self._covariance_regularization)

        # Initialize the mixture.
        mu, cov, weight = _initialize(y, **kwds)

        N, D = y.shape
        iterations = 1
        
        R, ll, message_length = _expectation(y, mu, cov, weight, **kwds)
        ll_dl = [(ll, message_length)]

        while True:

            M = weight.size
            best_perturbations = defaultdict(lambda: [np.inf])
                
            # Exhaustively split all components.
            for m in range(M):
                p = split_component(y, mu, cov, weight, R, m, **kwds)

                # Keep best split component.
                if p[-1] < best_perturbations["split"][-1]:
                    best_perturbations["split"] = [m] + list(p)
            
            if M > 1:
                # Exhaustively delete all components.
                for m in range(M):
                    p = delete_component(y, mu, cov, weight, R, m, **kwds)
                    
                    # Keep best deleted component.
                    if p[-1] < best_perturbations["delete"][-1]:
                        best_perturbations["delete"] = [m] + list(p)
                
                # Exhaustively merge all components.
                for m in range(M):
                    p = merge_component(y, mu, cov, weight, R, m, **kwds)

                    # Keep best merged component.
                    if p[-1] < best_perturbations["merge"][-1]:
                        best_perturbations["merge"] = [m] + list(p)

            # Get best perturbation.
            bop, bp = min(best_perturbations.items(), key=lambda x: x[1][-1])
            b_m, b_mu, b_cov, b_weight, b_R, b_meta, b_ml = bp

            logger.debug("Best operation: {} {}".format(bop, b_ml))

            if message_length > b_ml:
                # Set the new state as the best perturbation.
                iterations += 1
                message_length = b_ml
                mu, cov, weight, R = (b_mu, b_cov, b_weight, b_R)

            else:
                # None of the perturbations were better than what we had.
                break

        # TODO: a full_output response.
        meta = dict(message_length=message_length)
        return (mu, cov, weight, meta)
        
    fit = _fit_kasarapu_allison



    def search(self, y, **kwargs):

        kwds = dict(
            threshold=self._threshold, 
            max_em_iterations=self._max_em_iterations,
            covariance_type=self.covariance_type, 
            covariance_regularization=self._covariance_regularization)

        # Initialize the mixture.
        mu, cov, weight = _initialize(y, **kwds)

        N, D = y.shape
        iterations = 1
        
        R, ll, message_length = _expectation(y, mu, cov, weight, **kwds)
        ll_dl = [(ll, message_length)]

        while True:


            M = weight.size
            
            best_perturbations = defaultdict(lambda: [np.inf])
            logger.debug("K = {}, I = {}, ll = {}".format(M, message_length, ll))
            logger.debug("Smallest Neff = {} ({})".format(
                np.min(weight * y.shape[0]), np.argmin(weight * y.shape[0])))

            # Exhaustively split all components.
            for m in range(M):
                try:
                    p = split_component(y, mu, cov, weight, R, m, **kwds)
                except ValueError:
                    logger.debug("Failed to split component {}".format(m))
                    continue

                logger.debug("Split component {} of {}: {}".format(m, M, p[-1]))

                # If the split resulted in a maximum Neff of < 1, don't keep
                # this.
                if np.min(p[2] * y.shape[0]) < 1:
                    logger.debug("Ignoring mixture because Neff < 1")
                    continue

                self._mixture_predictors.append(p[-2]["predictors"])

                # Keep best split component.
                if p[-1] < best_perturbations["split"][-1]:
                    best_perturbations["split"] = [m] + list(p)
            
            if M > 1:
                # Exhaustively delete all components.
                for m in range(M):
                    try:
                        p = delete_component(y, mu, cov, weight, R, m, **kwds)
                    except ValueError:
                        logger.debug("Failed to delete component {}".format(m))
                        continue

                    logger.debug("Delete component {} of {}: {}".format(m, M, p[-1]))

                    if np.min(p[2] * y.shape[0]) < 1:
                        logger.debug("Ignoring mixture because Neff < 1")
                        continue


                    self._mixture_predictors.append(p[-2]["predictors"])

                    # Keep best deleted component.
                    if p[-1] < best_perturbations["delete"][-1]:
                        best_perturbations["delete"] = [m] + list(p)
                
                # Exhaustively merge all components.
                for m in range(M):
                    try:
                        p = merge_component(y, mu, cov, weight, R, m, **kwds)
                    
                    except ValueError:
                        logger.debug("Failed to delete component {}".format(m))
                        continue

                    if np.min(p[2] * y.shape[0]) < 1:
                        logger.debug("Ignoring mixture because Neff < 1")
                        continue


                    logger.debug("Merge component {} of {}: {}".format(m, M, p[-1]))

                    self._mixture_predictors.append(p[-2]["predictors"])

                    # Keep best merged component.
                    if p[-1] < best_perturbations["merge"][-1]:
                        best_perturbations["merge"] = [m] + list(p)

            # Get best perturbation.
            bop, bp = min(best_perturbations.items(), key=lambda x: x[1][-1])
            b_m, b_mu, b_cov, b_weight, b_R, b_meta, b_ml = bp

            # Consider a jump.
            jump = self._propose_jump(y, mu, cov, weight, message_length, ll)
            if jump:
                try:
                    p = jump_to_mixture(y, jump, **kwds)
                except ValueError:
                    logger.debug("Failed to jump to K = {}".format(jump))

                else:
                    self._mixture_predictors.append(p[-2]["predictors"])

                    if np.min(p[2] * y.shape[0]) < 1:
                        logger.debug("Ignoring mixture because Neff < 1")
                        continue


                    logger.debug("Jump to {} had I = {} [{}]".format(jump, p[-1],
                        "BETTER" if p[-1] < message_length else "WORSE"))

                    # Is a jump better than our best operation?
                    if p[-1] < b_ml:
                        # Accept the jump.
                        b_mu, b_cov, b_weight, b_R, b_meta, b_ml = p
                        b_op = "jump"

                        logger.debug("Accepted jump as best operation")

            logger.debug("Best operation: {} {}".format(bop, b_ml))

            if message_length > b_ml:
                # Set the new state as the best perturbation.
                iterations += 1
                message_length = b_ml
                ll = b_meta["log_likelihood"]
                mu, cov, weight, R = (b_mu, b_cov, b_weight, b_R)

            else:
                # None of the perturbations were better than what we had.
                break

        # Now cull things with zero members.
        """
        while True:
            del_index = np.where(weight * y.shape[0] < 1)[0]
            if len(del_index) == 0:
                break

            del_index = del_index[0]

            logger.debug("CULLING INDEX = {} because {}".format(del_index,
                weight[del_index] * y.shape[0]))
            logger.debug("State K = {} I = {} ll = {}".format(weight.size,
                message_length, ll))
            mu, cov, weight, R, meta, ml = delete_component(y, mu, cov, weight, R, del_index, **kwds)
            message_length = ml
            ll = meta["log_likelihood"]
        """
        assert np.min(weight * y.shape[0]) >= 1
        
        logger.debug("Ended on K = {}".format(weight.size))



        meta = dict(message_length=message_length, log_likelihood=-b_ml)
        return (mu, cov, weight, meta)


    def _propose_jump(self, y, mu, cov, weight, I, nll):

        # Search dK around the current position.
        predictors = np.array(self._mixture_predictors)
        if predictors.shape[0] < 2:
            return None

        N, D = y.shape
        offset = np.arange(self._predict_mixtures + 1)
        proposed_K = np.unique(np.clip(np.hstack([
            weight.size - offset,
            weight.size + offset
        ]), 1, N))

        # Predict future message lengths from our current state.
        pI, pI_scatter, pI_bound \
            = self._predict_message_length(proposed_K, y, mu, cov, weight, I, nll)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(proposed_K, pI, c="b")
        ax.plot(proposed_K, pI_bound, c="r")
        ax.fill_between(proposed_K, pI + pI_scatter, pI - pI_scatter, alpha=0.5,
            facecolor="b")
        ax.scatter(predictors.T[0], predictors.T[2], facecolor="k")
        ax.set_title("predicted mls from {}".format(weight.size))

        percent_scatter = 1.0 # MAGIC
        mixture_probability = 1e-3 # MAGIC
        reasonable_jump \
            = (abs(100 * pI_scatter/pI) < percent_scatter) \
            * (stats.norm(pI, pI_scatter).cdf(I) > mixture_probability)

        # Exclude places we have already been.
        been = lambda k: k in predictors.T[0]
        reasonable_jump *= np.array([not been(k) for k in proposed_K])

        if not any(reasonable_jump):
            logger.debug(
                "No reasonable jump from K = {} (tried from {} to {})"\
                .format(weight.size, proposed_K.min(), proposed_K.max()))
            return None

        else:
            logger.debug("Acceptable jumps from {}: {}".format(
                weight.size, proposed_K[reasonable_jump]))

        
            best_jump = proposed_K[reasonable_jump][np.argmin(pI_bound[reasonable_jump])] # TODO
            if abs(best_jump - weight.size) > 1:
                logger.debug("JUMPING to {}".format(best_jump))
                return best_jump

            else:
                logger.debug("Best jump was K = {} -- ignoring".format(best_jump))
                return None


    def _predict_message_length(self, proposed_K, y, mu, cov, weight, I, nll,
        lower_bound_sigma=5):

        
        predictors = np.array(self._mixture_predictors)

        slw_expectation, slw_variance, slw_upper \
            = self._approximate_sum_log_weights(proposed_K, weight, predictors=predictors)

        nll_mslogdetcov_expectation, nll_mslogdetcov_variance \
            = self._approximate_nllpslogdetcov(proposed_K, weight, predictors=predictors)

        slogdet = _slogdet(cov, self._covariance_type)

        N, D = y.shape

        current_K, D = mu.shape
        dK = proposed_K - current_K

        lnQ_ratio = np.log(
            _total_parameters(proposed_K, D, self._covariance_type) \
            / float(_total_parameters(current_K, D, self._covariance_type)))

        pI_expectation = I + dK * (
            (1 - D/2.0)*np.log(2) + 0.25 * (D*(D+3) + 2)*np.log(N/(2*np.pi))) \
            + 0.5 * (D*(D+3)/2.0 - 1) * (slw_expectation - np.sum(np.log(weight))) \
            - scipy.special.gammaln(proposed_K) + scipy.special.gammaln(current_K) \
            + 0.5 * lnQ_ratio \
            + (D + 2)/2.0 * (np.sum(slogdet)) \
            - nll + nll_mslogdetcov_expectation

        dI_scatter = (slw_variance + nll_mslogdetcov_variance)**0.5
        pI_lower_bound = I + dK * (
            (1 - D/2.0)*np.log(2) + 0.25 * (D*(D+3) + 2)*np.log(N/(2*np.pi))) \
            + 0.5 * (D*(D+3)/2.0 - 1) * (slw_upper - np.sum(np.log(weight))) \
            - scipy.special.gammaln(current_K + dK) + scipy.special.gammaln(current_K) \
            + 0.5 * lnQ_ratio \
            - (D + 2)/2.0 * (np.sum(slogdet)) \
            - nll + nll_mslogdetcov_expectation \
            - lower_bound_sigma * dI_scatter


        '''

        N, D = y.shape
        M = weight.size

        # I(M) = M\log{2} + constant
        I_m = M # [bits]

        # I(w) = \frac{(M - 1)}{2}\log{N} - \frac{1}{2}\sum_{j=1}^{M}\log{w_j} - (M - 1)!
        I_w = (M - 1) / 2.0 * np.log(N) \
            - 0.5 * np.sum(np.log(weight)) \
            - scipy.special.gammaln(M)

        I_w = I_w/np.log(2) # [bits]

        if covariance_type == "diag":
            cov_ = np.array([_ * np.eye(D) for _ in cov])
        else:
            # full
            cov_ = cov

        log_det_cov = np.log(np.linalg.det(cov_))
    
        log_F_m = 0.5 * D * (D + 3) * np.log(np.sum(responsibility, axis=1)) 
        log_F_m += -log_det_cov
        log_F_m += -(D * np.log(2) + (D + 1) * log_det_cov)

        
        AOM = 0.001 # MAGIC
        Il = nll - (D * N * np.log(AOM))
        Il = Il/np.log(2) # [bits]

        
        I_t = (0 + 0.5 * log_F_m)/np.log(2)
        sum_It = np.sum(I_t)

        num_free_params = (0.5 * D * (D+3) * M) + (M - 1)
        lattice = 0.5 * num_free_params * log_kappa(num_free_params) / np.log(2)

        part1 = I_m + I_w + np.sum(I_t) + lattice
        part2 = Il + (0.5 * num_free_params)/np.log(2)

        I = part1 + part2
        '''

        #print("Predicted change from {}: \n{}\n{}".format(weight.size, proposed_K, pI_expectation - I))
        result = (pI_expectation, dI_scatter, pI_lower_bound)
        return result




    def _approximate_sum_log_weights(self, K, weight, predictors=None):
        r"""
        Return an approximate expectation of the function:

        .. math:

            \sum_{k=1}^{K}\log{w_k}

        Where :math:`K` is the number of mixtures, and :math:`w` is a multinomial
        distribution. The approximating function is:

        .. math:

            \sum_{k=1}^{K}\log{w_k} \approx -K\log{K}

        :param K:
            The number of target Gaussian mixtures.
        """

        if predictors is None:
            predictors = np.array(self._mixture_predictors)

        k, slw = (predictors.T[0], predictors.T[1])

        k = np.sort(np.unique(predictors.T[0]))
        slw = np.ones(k.size)
        for i, ki in enumerate(k):
            match = (predictors.T[0] == ki)
            slw[i] = np.min(predictors.T[1][match])


        # Upper bound.
        upper_bound = lambda k, c=0: -k * np.log(k) + c

        #upper = -K * np.log(K)

        # Some expectation value.
        if 2 > len(k):
            # Don't provide an expectation value.
            expectation = upper_bound(K)
            variance \
                = np.abs(upper_bound(K**2) - upper_bound(K)**2)

        else:
            lower_values = [[k[0], slw[0]]]
            for k_, slw_ in zip(k[1:], slw[1:]):
                if k_ == lower_values[-1][0] and slw_ < lower_values[-1][1]:
                    lower_values[-1][1] = slw_
                elif k_ > lower_values[-1][0]:
                    lower_values.append([k_, slw_])
            lower_values = np.array(lower_values)

            function = lambda x, *p: -x * p[0] * np.log(x) + p[1]

            # Expectation, from the best that can be done.
            exp_params, exp_cov = op.curve_fit(
                function, lower_values.T[0], lower_values.T[1], p0=[1, 0])
            expectation = function(K, *exp_params)

            #exp_params, exp_cov = op.curve_fit(function, k, slw, p0=[1, 0])
            #expectation = function(K, *exp_params)

            variance = 0.0
            if np.isfinite(exp_cov).all():
                variance = np.var([
                    function(K, *draw) for draw in np.random.multivariate_normal(exp_params, exp_cov, size=30)],
                    axis=0)
            """
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_title("slogw K = {}".format(self.weight.size))
            ax.scatter(predictors.T[0], predictors.T[1], s=5, alpha=0.5,
                facecolor="k")

            ax.scatter(k, function(k, *exp_params), facecolor="b")
            ax.fill_between(K, expectation - variance**0.5, expectation + variance**0.5,
                alpha=0.5, facecolor="b")
            """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(predictors.T[0], predictors.T[1], facecolor="k")
        ax.plot(K, expectation, c="b")
        ax.fill_between(K, expectation - variance**0.5, expectation + variance**0.5,
            alpha=0.5, facecolor="b")
        ax.set_title("predicting sum log weights from {}".format(weight.size))
        ax.plot(K, upper_bound(K), c="r", lw=2)

        return (expectation, variance, upper_bound(K))



    def _approximate_nllpslogdetcov(self, target_K, weight, predictors=None):

        if predictors is None:
            predictors = np.array(self._mixture_predictors)

        # Get the lower values.
        x = np.sort(np.unique(predictors.T[0]))
        y = np.empty(x.size)
        yerr = np.empty(x.size)
        for i, xi in enumerate(x):
            match = (predictors.T[0] == xi)
            values = np.unique(np.round(predictors.T[-1][match], 2))
            y[i] = np.min(values)
            yerr[i] = np.std(values)

        yerr[(yerr == 0)] = np.max(yerr)

        
        function = lambda x, *p: np.polyval(p, x)

        # TODO: How to estimate variance in all this?
        op_params, op_cov = op.curve_fit(function, x, y, p0=np.zeros(3),
            sigma=yerr, absolute_sigma=True)

        expectation = function(target_K, *op_params)
        try:
            variance = np.var([function(target_K, *draw) \
                for draw in np.random.multivariate_normal(op_params, op_cov, size=30)],
                axis=0)
        except ValueError:
            variance = 0.0

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_title("slogdet K = {}".format(weight.size))

        ax.scatter(predictors.T[0], predictors.T[-1], s=5, alpha=0.5,
            facecolor="k")
        ax.scatter(x, y, facecolor="g", s=10)
        ax.errorbar(x, y, yerr)

        ax.plot(target_K, expectation, c="b")
        ax.fill_between(target_K, expectation - variance**0.5, expectation + variance**0.5,
            alpha=0.5, facecolor="b")
        
        return (expectation, variance)

def responsibility_matrix(y, mu, cov, weight, covariance_type, 
    full_output=False, **kwargs):
    r"""
    Return the responsibility matrix,

    .. math::

        r_{ij} = \frac{w_{j}f\left(x_i;\theta_j\right)}{\sum_{k=1}^{K}{w_k}f\left(x_i;\theta_k\right)}


    where :math:`r_{ij}` denotes the conditional probability of a datum
    :math:`x_i` belonging to the :math:`j`-th component. The effective 
    membership associated with each component is then given by

    .. math::

        n_j = \sum_{i=1}^{N}r_{ij}
        \textrm{and}
        \sum_{j=1}^{M}n_{j} = N


    where something.
    
    :param y:
        The data values, :math:`y`.

    :param mu:
        The mean values of the :math:`K` multivariate normal distributions.

    :param cov:
        The covariance matrices of the :math:`K` multivariate normal
        distributions. The shape of this array will depend on the 
        ``covariance_type``.

    :param weight:
        The current estimates of the relative mixing weight.

    :param full_output: [optional]
        If ``True``, return the responsibility matrix, and the log likelihood,
        which is evaluated for free (default: ``False``).

    :returns:
        The responsibility matrix. If ``full_output=True``, then the
        log likelihood (per observation) will also be returned.
    """

    precision_cholesky = _compute_precision_cholesky(cov, covariance_type)
    weighted_log_prob = np.log(weight) + \
        _estimate_log_gaussian_prob(y, mu, precision_cholesky, covariance_type)

    log_likelihood = scipy.misc.logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under="ignore"):
        log_responsibility = weighted_log_prob - log_likelihood[:, np.newaxis]

    responsibility = np.exp(log_responsibility).T
    
    return (responsibility, log_likelihood) if full_output else responsibility


def kullback_leibler_for_multivariate_normals(mu_a, cov_a, mu_b, cov_b):
    r"""
    Return the Kullback-Leibler distance from one multivariate normal
    distribution with mean :math:`\mu_a` and covariance :math:`\Sigma_a`,
    to another multivariate normal distribution with mean :math:`\mu_b` and 
    covariance matrix :math:`\Sigma_b`. The two distributions are assumed to 
    have the same number of dimensions, such that the Kullback-Leibler 
    distance is

    .. math::

        D_{\mathrm{KL}}\left(\mathcal{N}_{a}||\mathcal{N}_{b}\right) = 
            \frac{1}{2}\left(\mathrm{Tr}\left(\Sigma_{b}^{-1}\Sigma_{a}\right) + \left(\mu_{b}-\mu_{a}\right)^\top\Sigma_{b}^{-1}\left(\mu_{b} - \mu_{a}\right) - k + \ln{\left(\frac{\det{\Sigma_{b}}}{\det{\Sigma_{a}}}\right)}\right)


    where :math:`k` is the number of dimensions and the resulting distance is 
    given in units of nats.

    .. warning::

        It is important to remember that 
        :math:`D_{\mathrm{KL}}\left(\mathcal{N}_{a}||\mathcal{N}_{b}\right) \neq D_{\mathrm{KL}}\left(\mathcal{N}_{b}||\mathcal{N}_{a}\right)`.


    :param mu_a:
        The mean of the first multivariate normal distribution.

    :param cov_a:
        The covariance matrix of the first multivariate normal distribution.

    :param mu_b:
        The mean of the second multivariate normal distribution.

    :param cov_b:
        The covariance matrix of the second multivariate normal distribution.
    
    :returns:
        The Kullback-Leibler distance from distribution :math:`a` to :math:`b`
        in units of nats. Dividing the result by :math:`\log_{e}2` will give
        the distance in units of bits.
    """

    if len(cov_a.shape) == 1:
        cov_a = cov_a * np.eye(cov_a.size)

    if len(cov_b.shape) == 1:
        cov_b = cov_b * np.eye(cov_b.size)

    U, S, V = np.linalg.svd(cov_a)
    Ca_inv = np.dot(np.dot(V.T, np.linalg.inv(np.diag(S))), U.T)

    U, S, V = np.linalg.svd(cov_b)
    Cb_inv = np.dot(np.dot(V.T, np.linalg.inv(np.diag(S))), U.T)

    k = mu_a.size

    offset = mu_b - mu_a
    return 0.5 * np.sum([
          np.trace(np.dot(Ca_inv, cov_b)),
        + np.dot(offset.T, np.dot(Cb_inv, offset)),
        - k,
        + np.log(np.linalg.det(cov_b)/np.linalg.det(cov_a))
    ])


def _parameters_per_mixture(D, covariance_type):
    r"""
    Return the number of parameters per Gaussian component, given the number 
    of observed dimensions and the covariance type.

    :param D:
        The number of dimensions per data point.

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a free covariance matrix, or
        `diag` for a diagonal covariance matrix.

    :returns:
        The number of parameters required to fully specify the multivariate
        mean and covariance matrix of a :math:`D`-dimensional Gaussian.
    """

    if covariance_type == "full":
        return int(D + D*(D + 1)/2.0)
    elif covariance_type == "diag":
        return 2 * D
    else:
        raise ValueError("unknown covariance type '{}'".format(covariance_type))


def _initialize(y, covariance_type, covariance_regularization, **kwargs):
    r"""
    Return initial estimates of the parameters.

    :param y:
        The data values, :math:`y`.

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a free covariance matrix, or
        `diag` for a diagonal covariance matrix.

    :param covariance_regularization:
        Regularization strength to add to the diagonal of covariance matrices.


    :returns:
        A three-length tuple containing the initial (multivariate) mean,
        the covariance matrix, and the relative weight.
    """

    # If you *really* know what you're doing, then you can give your own.
    if kwargs.get("__initialize", None) is not None:
        return kwargs.pop("__initialize")

    weight = np.ones((1, 1))
    N, D = y.shape
    mean = np.mean(y, axis=0).reshape((1, -1))

    cov = _estimate_covariance_matrix(y, np.ones((1, N)), mean,
        covariance_type, covariance_regularization)

    return (mean, cov, weight)
    

def _expectation(y, mu, cov, weight, **kwargs):
    r"""
    Perform the expectation step of the expectation-maximization algorithm.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The current best estimates of the (multivariate) means of the :math:`K`
        components.

    :param cov:
        The current best estimates of the covariance matrices of the :math:`K`
        components.

    :param weight:
        The current best estimates of the relative weight of all :math:`K`
        components.

    :param N_component_pars:
        The number of parameters required to specify the mean and covariance
        matrix of a single Gaussian component.

    :returns:
        A three-length tuple containing the responsibility matrix,
        the log likelihood, and the change in message length.
    """

    responsibility, log_likelihood = responsibility_matrix(
        y, mu, cov, weight, full_output=True, **kwargs)

    nll = -np.sum(log_likelihood)

    I = _message_length(y, mu, cov, weight, responsibility, nll, **kwargs)
    
    try:
        kwargs["callback"]("expectation.state", responsibility, log_likelihood, I)

    except KeyError:
        None

    except:
        raise

    return (responsibility, nll, I)


def log_kappa(D):

    cd = -0.5 * D * np.log(2 * np.pi) + 0.5 * np.log(D * np.pi)
    return -1 + 2 * cd/D



def _message_length(y, mu, cov, weight, responsibility, nll,
    covariance_type, eps=0.10, dofail=False, **kwargs):

    # THIS IS SO BAD

    N, D = y.shape
    M = weight.size

    # I(M) = M\log{2} + constant
    I_m = M # [bits]

    # I(w) = \frac{(M - 1)}{2}\log{N} - \frac{1}{2}\sum_{j=1}^{M}\log{w_j} - (M - 1)!
    I_w = (M - 1) / 2.0 * np.log(N) \
        - 0.5 * np.sum(np.log(weight)) \
        - scipy.special.gammaln(M)


    I_w = I_w/np.log(2) # [bits]

    if D == 1:
        log_F_m = np.log(2) + (2 * np.log(N)) - 4 * np.log(cov.flatten()[0]**0.5)
        raise UnsureError

    else:
        if covariance_type == "diag":
            cov_ = np.array([_ * np.eye(D) for _ in cov])
        else:
            # full
            cov_ = cov

        _, log_det_cov = np.linalg.slogdet(cov_)#np.log(np.linalg.det(cov_))
    
        log_F_m = 0.5 * D * (D + 3) * np.log(np.sum(responsibility, axis=1)) 
        log_F_m += -log_det_cov
        log_F_m += -(D * np.log(2) + (D + 1) * log_det_cov)

        
    # TODO: No prior on h(theta).. thus -\sum_{j=1}^{M}\log{h\left(\theta_j\right)} = 0

    # TODO: bother about including this? -N * D * np.log(eps)
    

    AOM = 0.001 # MAGIC
    Il = nll - (D * N * np.log(AOM))
    Il = Il/np.log(2) # [bits]

    
    log_prior = 0


    I_t = (log_prior + 0.5 * log_F_m)/np.log(2)
    sum_It = np.sum(I_t)


    num_free_params = (0.5 * D * (D+3) * M) + (M - 1)
    lattice = 0.5 * num_free_params * log_kappa(num_free_params) / np.log(2)


    part1 = I_m + I_w + np.sum(I_t) + lattice
    part2 = Il + (0.5 * num_free_params)/np.log(2)

    I = part1 + part2

    assert I_w >= -0.5 # prevent triggers on underflow

    assert I > 0

    if dofail:
    
        print(I_m, I_w, np.sum(I_t), lattice, Il)
        print(I_t)
        print(part1, part2)

        raise a

    return I


def _compute_precision_cholesky(covariances, covariance_type):
    r"""
    Compute the Cholesky decomposition of the precision of the covariance
    matrices provided.

    :param covariances:
        An array of covariance matrices.

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a free covariance matrix, or
        `diag` for a diagonal covariance matrix.
    """

    singular_matrix_error = "Failed to do Cholesky decomposition"

    if covariance_type in "full":
        M, D, _ = covariances.shape

        cholesky_precision = np.empty((M, D, D))
        for m, covariance in enumerate(covariances):
            try:
                cholesky_cov = scipy.linalg.cholesky(covariance, lower=True) 
            except scipy.linalg.LinAlgError:
                raise ValueError(singular_matrix_error)


            cholesky_precision[m] = scipy.linalg.solve_triangular(
                cholesky_cov, np.eye(D), lower=True).T

    elif covariance_type in "diag":
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(singular_matrix_error)

        cholesky_precision = covariances**(-0.5)

    else:
        raise NotImplementedError("nope")

    return cholesky_precision



def _estimate_covariance_matrix_full(y, responsibility, mean, 
    covariance_regularization=0):

    N, D = y.shape
    M, N = responsibility.shape

    membership = np.sum(responsibility, axis=1)

    I = np.eye(D)
    cov = np.empty((M, D, D))
    for m, (mu, rm, nm) in enumerate(zip(mean, responsibility, membership)):

        diff = y - mu
        denominator = nm - 1 if nm > 1 else nm

        cov[m] = np.dot(rm * diff.T, diff) / denominator \
               + covariance_regularization * I

    return cov


def _estimate_covariance_matrix(y, responsibility, mean, covariance_type,
    covariance_regularization):

    available = {
        "full": _estimate_covariance_matrix_full,
        "diag": _estimate_covariance_matrix_diag
    }

    try:
        function = available[covariance_type]

    except KeyError:
        raise ValueError("unknown covariance type")

    return function(y, responsibility, mean, covariance_regularization)

def _estimate_covariance_matrix_diag(y, responsibility, mean, 
    covariance_regularization=0):

    N, D = y.shape
    M, N = responsibility.shape

    denominator = np.sum(responsibility, axis=1)
    denominator[denominator > 1] = denominator[denominator > 1] - 1

    membership = np.sum(responsibility, axis=1)

    I = np.eye(D)
    cov = np.empty((M, D))
    for m, (mu, rm, nm) in enumerate(zip(mean, responsibility, membership)):

        diff = y - mu
        denominator = nm - 1 if nm > 1 else nm

        cov[m] = np.dot(rm, diff**2) / denominator + covariance_regularization

    return cov

    


def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.
    Parameters
    ----------
    matrix_chol : array-like,
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    n_features : int
        Number of features.
    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
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


def _estimate_log_gaussian_prob(X, means, precision_cholesky, covariance_type):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precision_cholesky, covariance_type, n_features)

    if covariance_type in 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precision_cholesky)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type in 'diag':
        precisions = precision_cholesky**2
        log_prob = (np.sum((means ** 2 * precisions), 1) - 2.0 * np.dot(X, (means * precisions).T) + np.dot(X**2, precisions.T))

    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


def _maximization(y, mu, cov, weight, responsibility, parent_responsibility=1,
    **kwargs):
    r"""
    Perform the maximization step of the expectation-maximization algorithm
    on all components.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current best estimates of the relative weight of all :math:`K`
        components.

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.
    
    :param parent_responsibility: [optional]
        An array of length :math:`N` giving the parent component 
        responsibilities (default: ``1``). Only useful if the maximization
        step is to be performed on sub-mixtures with parent responsibilities.

    :returns:
        A three length tuple containing the updated multivariate mean values,
        the updated covariance matrices, and the updated mixture weights. 
    """

    M = weight.size 
    N, D = y.shape
    
    # Update the weights.
    effective_membership = np.sum(responsibility, axis=1)
    new_weight = (effective_membership + 0.5)/(N + M/2.0)

    w_responsibility = parent_responsibility * responsibility
    w_effective_membership = np.sum(w_responsibility, axis=1)

    new_mu = np.zeros_like(mu)
    new_cov = np.zeros_like(cov)
    for m in range(M):
        new_mu[m] = np.sum(w_responsibility[m] * y.T, axis=1) \
                  / w_effective_membership[m]

    new_cov = _estimate_covariance_matrix(y, responsibility, new_mu,
        kwargs["covariance_type"], kwargs["covariance_regularization"])

    state = (new_mu, new_cov, new_weight)

    assert np.all(np.isfinite(new_mu))
    assert np.all(np.isfinite(new_cov))
    assert np.all(np.isfinite(new_weight))

    return state 




def _expectation_maximization(y, mu, cov, weight, responsibility=None, **kwargs):
    r"""
    Run the expectation-maximization algorithm on the current set of
    multivariate Gaussian mixtures.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current estimates of the relative mixing weight.

    :param responsibility: [optional]
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component. If ``None`` is given
        then the responsibility matrix will be calculated in the first
        expectation step.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :param threshold: [optional]
        The relative improvement in log likelihood required before stopping
        an expectation-maximization step (default: ``1e-5``).

    :param max_em_iterations: [optional]
        The maximum number of iterations to run per expectation-maximization
        loop (default: ``10000``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """   

    M = weight.size
    N, D = y.shape
    
    # Calculate log-likelihood and initial expectation step.
    _init_responsibility, ll, dl = _expectation(y, mu, cov, weight, **kwargs)

    if responsibility is None:
        responsibility = _init_responsibility

    iterations = 1
    ll_dl = [(ll, dl)]

    while True:

        # Perform the maximization step.
        mu, cov, weight \
            = _maximization(y, mu, cov, weight, responsibility, **kwargs)

        # Run the expectation step.
        responsibility, ll, dl \
            = _expectation(y, mu, cov, weight, **kwargs)

        # Check for convergence.
        prev_ll, prev_dl = ll_dl[-1]
        relative_delta_message_length = np.abs((ll - prev_ll)/prev_ll)
        ll_dl.append([ll, dl])
        iterations += 1

        assert np.isfinite(relative_delta_message_length)

        if relative_delta_message_length <= kwargs["threshold"] \
        or iterations >= kwargs["max_em_iterations"]:
            break

    meta = dict(warnflag=iterations >= kwargs["max_em_iterations"], log_likelihood=ll)
    if meta["warnflag"]:
        logger.warn("Maximum number of E-M iterations reached ({}) {}".format(
            kwargs["max_em_iterations"], kwargs.get("_warn_context", "")))

    slogdet = np.sum(_slogdet(cov, kwargs["covariance_type"]))
    meta["predictors"] = [
        weight.size,
        np.sum(np.log(weight)),
        dl,
        ll, # THIS IS ACTUALLY THE negative LOG LIKELIHOOD WARNING FUCK
        ll - (D + 2)/2.0 * slogdet
    ]

    return (mu, cov, weight, responsibility, meta, dl)


def _total_parameters(K, D, covariance_type):
    r"""
    Return the total number of model parameters :math:`Q`, if a full 
    covariance matrix structure is assumed.

    .. math:

        Q = \frac{K}{2}\left[D(D+3) + 2\right] - 1


    :param K:
        The number of Gaussian mixtures.

    :param D:
        The dimensionality of the data.

    :returns:
        The total number of model parameters, :math:`Q`.
    """
    return (0.5 * D * (D + 3) * K) + (K - 1)


def _svd(covariance, covariance_type):

    if covariance_type == "full":
        return np.linalg.svd(covariance)

    elif covariance_type == "diag":
        return np.linalg.svd(covariance * np.eye(covariance.size))

    else:
        raise ValueError("unknown covariance type")

def _slogdet(covariance, covariance_type):

    if covariance_type == "full":
        sign, slogdet = np.linalg.slogdet(covariance)
        assert np.all(sign == 1)
        return slogdet

    elif covariance_type == "diag":
        K, D = covariance.shape
        cov = np.array([_ * np.eye(D) for _ in covariance])
        sign, slogdet = np.linalg.slogdet(cov)

        assert np.all(sign == 1)
        return slogdet

    else:
        raise ValueError("unknown covariance type")



def jump_to_mixture(y, K, **kwargs):
    
    # Initialize with kmeans++
    row_norms = kmeans.row_norms(y, squared=True)

    mu = kmeans._k_init(y, K, row_norms, kmeans.check_random_state(None))

    distance = np.sum((y[:, :, None] - mu.T)**2, axis=1).T

    N, D = y.shape
    responsibility = np.zeros((K, N))
    responsibility[np.argmin(distance, axis=0), np.arange(N)] = 1.0

    weight = responsibility.sum(axis=1)/N

    cov = _estimate_covariance_matrix(
        y, responsibility, mu,
        kwargs["covariance_type"], kwargs["covariance_regularization"])

    # Run E-M
    return _expectation_maximization(y, mu, cov, weight, responsibility, 
        **kwargs)


def split_component(y, mu, cov, weight, responsibility, index, **kwargs):
    r"""
    Split a component from the current mixture and determine the new optimal
    state.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current estimates of the relative mixing weight.

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be split.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """

    # TODO: Current implementation only allows for a component to be split
    #       into two sub-components

    #logger.debug("Splitting component {} of {}".format(index, weight.size))

    M = weight.size
    N, D = y.shape
    
    # Compute the direction of maximum variance of the parent component, and
    # locate two points which are one standard deviation away on either side.
    U, S, V = _svd(cov[index], kwargs["covariance_type"])

    child_mu = mu[index] - np.vstack([+V[0], -V[0]]) * S[0]**0.5

    assert np.all(np.isfinite(child_mu))

    # Responsibilities are initialized by allocating the data points to the 
    # closest of the two means.
    distance = np.vstack([
        np.sum((y - child_mu[0])**2, axis=1),
        np.sum((y - child_mu[1])**2, axis=1)
    ])
    
    child_responsibility = np.zeros((2, N))
    child_responsibility[np.argmin(distance, axis=0), np.arange(N)] = 1.0

    # Calculate the child covariance matrices.
    child_cov = _estimate_covariance_matrix(y, child_responsibility, child_mu,
        kwargs["covariance_type"], kwargs["covariance_regularization"])

    child_effective_membership = np.sum(child_responsibility, axis=1)    
    child_weight = child_effective_membership.T/child_effective_membership.sum()

    # We will need these later.
    parent_weight = weight[index]
    parent_responsibility = responsibility[index]

    """
    # Run expectation-maximization on the child mixtures.
    child_mu, child_cov, child_weight, child_responsibility, meta, dl = \
        _expectation_maximization(y, child_mu, child_cov, child_weight, 
            responsibility=child_responsibility, 
            parent_responsibility=parent_responsibility,
            covariance_type=covariance_type, **kwargs)
    """

    # After the chld mixture is locally optimized, we need to integrate it
    # with the untouched M - 1 components to result in a M + 1 component
    # mixture M'.

    # An E-M is finally carried out on the combined M + 1 components to
    # estimate the parameters of M' and result in an optimized 
    # (M + 1)-component mixture.

    # Update the component weights.
    # Note that the child A mixture will remain in index `index`, and the
    # child B mixture will be appended to the end.

    if M > 1:

        # Integrate the M + 1 components and run expectation-maximization
        weight = np.hstack([weight, [parent_weight * child_weight[1]]])
        weight[index] = parent_weight * child_weight[0]

        responsibility = np.vstack([responsibility, 
            [parent_responsibility * child_responsibility[1]]])
        responsibility[index] = parent_responsibility * child_responsibility[0]
        
        mu = np.vstack([mu, [child_mu[1]]])
        mu[index] = child_mu[0]

        cov = np.vstack([cov, [child_cov[1]]])
        cov[index] = child_cov[0]

        mu, cov, weight, responsibility, meta, ml = _expectation_maximization(
            y, mu, cov, weight, responsibility, **kwargs)


    else:
        # Simple case where we don't have to re-run E-M because there was only
        # one component to split.
        child_mu, child_cov, child_weight, child_responsibility, meta, ml = \
            _expectation_maximization(y, child_mu, child_cov, child_weight, 
            responsibility=child_responsibility, 
            parent_responsibility=parent_responsibility, **kwargs)

        mu, cov, weight, responsibility \
            = (child_mu, child_cov, child_weight, child_responsibility)

    return (mu, cov, weight, responsibility, meta, ml)


def delete_component(y, mu, cov, weight, responsibility, index, **kwargs):
    r"""
    Delete a component from the mixture, and return the new optimal state.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current estimates of the relative mixing weight.

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be deleted.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """

    #logger.debug("Deleting component {} of {}".format(index, weight.size))

    # Create new component weights.
    parent_weight = weight[index]
    parent_responsibility = responsibility[index]

    new_mu = np.delete(mu, index, axis=0)
    new_cov = np.delete(cov, index, axis=0)
    
    # Eq. 54-55
    new_weight = np.clip(
        np.delete(weight, index, axis=0)/(1-parent_weight),
        0, 1)
    
    # Calculate the new responsibility safely.
    new_responsibility = np.delete(responsibility, index, axis=0) / (1 - parent_responsibility)

    # The non-finite values occur when one data point was wholly assigned
    # responsibility to the component being deleted.
    reassigned = np.where(~np.all(np.isfinite(new_responsibility), axis=0))[0]

    # L2 distance.
    distance = np.sum((y[:, :, None] - new_mu.T)**2, axis=1).T
    inv_norm_distance = np.sum(distance, axis=0)/distance
    norm_distance = inv_norm_distance/np.sum(inv_norm_distance, axis=0)


    #closest = np.argmin(distance, axis=0)

    #if weight.size > 5 and len(reassigned) > 0:
    #    raise a
    # TODO: calculate as normalized distance?
    new_responsibility[:, reassigned] = norm_distance[:, reassigned]

    new_weight = new_responsibility.sum(axis=1)/y.shape[0]

    #new_responsibility[closest[reassigned], reassigned] = 1.0

    #if len(reassigned) > 0:
    #    raise a
    assert np.all(np.isfinite(new_responsibility))
    #foo = new_responsibility.copy()
    #new_responsibility[~np.isfinite(new_responsibility)] = 0.0

    assert np.all(np.isfinite(new_responsibility))
    assert np.all(np.isfinite(new_weight))

    
    # Run expectation-maximizaton on the perturbed mixtures. 
    return _expectation_maximization(y, new_mu, new_cov, new_weight, 
        new_responsibility, **kwargs)


def merge_component(y, mu, cov, weight, responsibility, index, **kwargs):
    r"""
    Merge a component from the mixture with its "closest" component, as
    judged by the Kullback-Leibler distance.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current estimates of the relative mixing weight.

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be deleted.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """

    # Calculate the Kullback-Leibler distance to the other distributions.
    D_kl = np.inf * np.ones(weight.size)
    for m in range(weight.size):
        if m == index: continue
        D_kl[m] = kullback_leibler_for_multivariate_normals(
            mu[index], cov[index], mu[m], cov[m])

    a_index, b_index = (index, np.nanargmin(D_kl))

    logger.debug("Merging component {} (of {}) with {}".format(
        a_index, weight.size, b_index))


    # Initialize.
    weight_k = np.sum(weight[[a_index, b_index]])
    responsibility_k = np.sum(responsibility[[a_index, b_index]], axis=0)
    effective_membership_k = np.sum(responsibility_k)

    mu_k = np.sum(responsibility_k * y.T, axis=1) / effective_membership_k
    cov_k = _estimate_covariance_matrix(
        y, np.atleast_2d(responsibility_k), np.atleast_2d(mu_k), 
        kwargs["covariance_type"], kwargs["covariance_regularization"])

    # Delete the b-th component.
    del_index = np.max([a_index, b_index])
    keep_index = np.min([a_index, b_index])

    new_mu = np.delete(mu, del_index, axis=0)
    new_cov = np.delete(cov, del_index, axis=0)
    new_weight = np.delete(weight, del_index, axis=0)
    new_responsibility = np.delete(responsibility, del_index, axis=0)

    new_mu[keep_index] = mu_k
    new_cov[keep_index] = cov_k
    new_weight[keep_index] = weight_k
    new_responsibility[keep_index] = responsibility_k

    # Calculate log-likelihood.
    return _expectation_maximization(y, new_mu, new_cov, new_weight,
        responsibility=new_responsibility,  **kwargs)