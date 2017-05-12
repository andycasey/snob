
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
import scipy.stats as stats
import scipy.optimize as op
from collections import defaultdict
from sklearn.cluster import k_means_ as kmeans

logger = logging.getLogger(__name__)



def _total_parameters(K, D):
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



def _responsibility_matrix(y, mean, covariance, weight, covariance_type):
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

    precision = _compute_precision_cholesky(covariance, covariance_type)
    weighted_log_prob = np.log(weight) + \
        _estimate_log_gaussian_prob(y, mean, precision, covariance_type)

    log_likelihood = scipy.misc.logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under="ignore"):
        log_responsibility = weighted_log_prob - log_likelihood[:, np.newaxis]

    responsibility = np.exp(log_responsibility).T
    
    return (responsibility, log_likelihood)




class BaseGaussianMixture(object):

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

    def __init__(self, covariance_type="full", covariance_regularization=0, 
        mixture_probability=1e-3, percent_scatter=1, predict_mixtures=3,
        threshold=1e-3, max_em_iterations=10000, **kwargs):

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
        self._mixture_probability = mixture_probability
        self._percent_scatter = percent_scatter
        self._predict_mixtures = predict_mixtures
        self._max_em_iterations = max_em_iterations
        self._covariance_type = covariance_type
        self._covariance_regularization = covariance_regularization
        
        return None

    @property
    def mean(self):
        r""" Return the multivariate means of the Gaussian mixtures. """
        return self._mean

    @property
    def covariance(self):
        r""" Return the covariance matrices of the Gaussian mixtures. """
        return self._covariance

    @property
    def weight(self):
        r""" Return the relative weights of the Gaussian mixtures. """
        return self._weight


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


    def _expectation(self, y, **kwargs):
        r"""
        Perform the expectation step of the expectation-maximization algorithm.

        :param y:
            The data values, :math:`y`.

        :returns:
            A three-length tuple containing the responsibility matrix,
            the  log likelihood, and the change in message length.
        """

        responsibility, log_likelihood = _responsibility_matrix(
            y, self.mean, self.covariance, self.weight, self.covariance_type)

        ll = np.sum(log_likelihood)

        I = _message_length(y, self.mean, self.covariance, self.weight,
            responsibility, -ll, self.covariance_type,
            **kwargs)

        return (responsibility, log_likelihood, I)


    def _maximization(self, y, responsibility, parent_responsibility=1,
        **kwargs):
        r"""
        Perform the maximization step of the expectation-maximization 
        algorithm.

        :param y:
            The data values, :math:`y`.
            # TODO

        :param responsibility:
            The responsibility matrix for all :math:`N` observations being
            partially assigned to each :math:`K` component.
            # TODO
        """

        K = self.weight.size 
        N, D = y.shape

        # Update the weights.
        effective_membership = np.sum(responsibility, axis=1)
        weight = (effective_membership + 0.5)/(N + K/2.0)

        w_responsibility = parent_responsibility * responsibility
        w_effective_membership = np.sum(w_responsibility, axis=1)

        mean = np.empty(self.mean.shape)
        for k, (R, Nk) in enumerate(zip(w_responsibility, w_effective_membership)):
            mean[k] = np.sum(R * y.T, axis=1) / Nk

        # TODO: Use parent responsibility when initializing?
        covariance = _estimate_covariance_matrix(y, responsibility, mean,
            self.covariance_type, self.covariance_regularization)

        # TODO: callback?
        return self.set_parameters(
            weight=weight, mean=mean, covariance=covariance)


    def _expectation_maximization(self, y, responsibility=None, **kwargs):
        r"""
        Run the expectation-maximization algorithm on the current set of
        multivariate Gaussian mixtures.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is the
            number of dimensions per observation.

        :param responsibility: [optional]
            The responsibility matrix for all :math:`N` observations being
            partially assigned to each :math:`K` component. If ``None`` is given
            then the responsibility matrix will be calculated in the first
            expectation step.
        """   

        # Calculate log-likelihood and initial expectation step.
        __init_responsibility, ll, dl = self._expectation(y, **kwargs)
        if responsibility is None:
            responsibility = __init_responsibility

        ll_dl = [(ll.sum(), dl)]

        meta = dict(warnflag=False)
        for iteration in range(self.max_em_iterations):

            # M-step.
            self._maximization(y, responsibility, **kwargs)

            # E-step.
            responsibility, ll, dl = self._expectation(y, **kwargs)

            # Check for convergence.
            lls = ll.sum()
            prev_ll, prev_dl = ll_dl[-1]
            change = (lls - prev_ll)/prev_ll
            ll_dl.append([lls, dl])

            #print("E-M", iteration, change, self.threshold)

            if abs(change) <= self.threshold:
                break

        else:
            meta.update(warnflag=True)
            logger.warn("Maximum number of E-M iterations reached ({})"\
                .format(self.max_em_iterations))

        meta.update(log_likelihood=lls, message_length=dl)

        return (responsibility, meta)


    @property
    def parameters(self):
        return dict([(k, getattr(self, k, None)) for k in self.parameter_names])

    def set_parameters(self, **kwargs):
        r"""
        Set specific parameters.
        """

        invalid_params = set(self.parameter_names).difference(kwargs.keys())
        if invalid_params:
            raise ValueError(
                "unknown parameters: {}".format(", ".join(invalid_params)))            
        
        for parameter_name, value in kwargs.items():
            setattr(self, "_{}".format(parameter_name), value)

        return kwargs


class GaussianMixture(BaseGaussianMixture):

    def __init__(self, **kwargs):
        super(GaussianMixture, self).__init__(**kwargs)

        # For predictions.
        self._proposed_mixtures = []

        # Store the following summary pieces of information about mixtures.
        # (1) Sum of the log of the determinant of the covariance matrices.
        # (2) The sum of the log-likelihood.
        # (3) The sum of the log of the weights.

        # Do we want this from each E-M step, or all steps?
        # (K, sum_log_weights, sum_log_likelihood, sum_log_det_covariances)
        self._mixture_predictors = []
        

    def _optimize_split_mixture(self, y, responsibility, component_index):
        r"""
        Split a component from the current mixture, and run partial 
        expectation-maximization algorithm on the split component.

        """
        U, S, V = _svd(self.covariance[component_index], self.covariance_type)

        split_mean = self.mean[component_index] \
                    + np.vstack([+V[0], -V[0]]) * S[0]**0.5

        # Responsibilities are initialized by allocating the data points to 
        # the closest of the two means.
        distance = np.sum((y[:, :, None] - split_mean.T)**2, axis=1).T

        N, D = y.shape
        split_responsibility = np.zeros((2, N))
        split_responsibility[np.argmin(distance, axis=0), np.arange(N)] = 1.0

        # Calculate the child covariance matrices.
        split_covariance = _estimate_covariance_matrix(
            y, split_responsibility, split_mean,
            self.covariance_type, self.covariance_regularization)

        split_effective_membership = np.sum(split_responsibility, axis=1)    
        split_weight = split_effective_membership.T \
                     / np.sum(split_effective_membership)

        # Integrate the split components with the existing mixture.
        parent_weight = self.weight[component_index]
        parent_responsibility = responsibility[component_index]

        mixture = self.__class__(
            threshold=self.threshold,
            covariance_type=self.covariance_type,
            max_em_iterations=self.max_em_iterations,
            covariance_regularization=self.covariance_regularization)

        # Initialize it.
        mixture.set_parameters(mean=split_mean, weight=split_weight,
            covariance=split_covariance)

        # Run E-M on the partial mixture.
        R, meta = mixture._expectation_maximization(
            y, parent_responsibility=responsibility[component_index])

        if self.weight.size > 1:
            # Integrate the partial mixture with the full mixture.
            weight = np.hstack([self.weight, 
                [parent_weight * mixture.weight[1]]])
            weight[component_index] = parent_weight * mixture.weight[0]

            mean = np.vstack([self.mean, [mixture.mean[1]]])
            mean[component_index] = mixture.mean[0]

            covariance = np.vstack([self.covariance, [mixture.covariance[1]]])
            covariance[component_index] = mixture.covariance[0]

            responsibility = np.vstack([responsibility,
                [parent_responsibility * R[1]]])
            responsibility[component_index] = parent_responsibility * R[0]

            mixture.set_parameters(
                mean=mean, covariance=covariance, weight=weight)

            R, meta = mixture._expectation_maximization(
                y, responsibility=responsibility)

        # Store the mixture.
        slogdet = np.sum(np.log(np.linalg.det(mixture.covariance)))
        self._proposed_mixtures.append(mixture)
        self._mixture_predictors.append([
            mixture.weight.size,
            np.sum(np.log(mixture.weight)),
            meta["log_likelihood"],
            slogdet,
            -meta["log_likelihood"] + (D+2)/2.0 * slogdet
        ])
        # TODO: Remove predictors that  we don't use.
        #self._slogs.append(np.linalg.det(mixture.covariance))

        return (len(self._proposed_mixtures) - 1, R, meta)

        # Run

        kwds = dict(
            threshold=self._threshold,
            max_em_iterations=self._max_em_iterations,
            covariance_type=self._covariance_type,
            covariance_regularization=self._covariance_regularization)

        # Run E-M on the split mixture, keeping all else fixed.
        #(dict(mean=mu, covariance=cov, weight=weight), responsibility, meta, dl)
        params, R, meta, dl = _expectation_maximization(y, split_mean, split_covariance,
            split_weight, responsibility=split_responsibility,
            parent_responsibility=parent_responsibility,
            **kwds)


        if self.weight.size > 1:

            # Integrate the child mixtures back.
            weight = np.hstack([self.weight, [parent_weight * params["weight"][1]]])
            weight[component_index] = parent_weight * params["weight"][0]

            mean = np.vstack([self.mean, [params["mean"][1]]])
            mean[component_index] = params["mean"][0]

            covariance = np.vstack([self.covariance, [params["covariance"][1]]])
            covariance[component_index] = params["covariance"][0]

            responsibility = np.vstack([responsibility, 
                [parent_responsibility * R[1]]])
            responsibility[component_index] \
                = parent_responsibility * R[0]

            return _expectation_maximization(y, mean, covariance, weight,
                responsibility=responsibility, **kwds)


        else:
            return (params, R, meta, dl)



    def _initialize_parameters(self, y, **kwargs):
        r"""
        Return initial estimates of the parameters.

        :param y:
            The data values, :math:`y`.
            # TODO COMMON DOCS
        """

        # If you *really* know what you're doing, then you can give your own.
        if kwargs.get("__initialize", None) is not None:
            logger.warn("Using specified initialization point.")
            return self.set_parameters(**kwargs.pop("__initialize"))
        
        weight = np.ones(1)
        mean = np.mean(y, axis=0).reshape((1, -1))

        covariance = _estimate_covariance_matrix(y, np.ones((1, 1)), mean,
            self.covariance_type, self.covariance_regularization)

        # Set parameters.
        return self.set_parameters(
            weight=weight, mean=mean, covariance=covariance)



    def _predict_message_length_change(self, K, N, lower_bound_sigma=5):
        r"""
        Predict the minimum message length of a target mixture of K Gaussian
        distributions, where K is an integer larger than the current mixture.

        :param K:
            The target number of Gaussian distributions. This must be an
            integer value larger than the current number of Gaussian mixtures.

        :returns:
            A pdf of some description. #TODO #YOLO
        """

        current_K, D = self.mean.shape
        #K = current_K + 1 if K is None else int(K)
        K = np.atleast_1d(K)
        if np.any(current_K >= K):
            raise ValueError(
                "the target K mixture must contain more Gaussians than the "\
                "current mixture ({} > {})".format(K, current_K))

        predictors = np.array(self._mixture_predictors)
        kwds = dict(target_K=K, predictors=predictors)

        dK = K - current_K


        slw_expectation, slw_variance, slw_upper \
            = self._approximate_sum_log_weights(**kwds)

        # Now approximate the sum of the negative log-likelihood, minus the
        # sum of the log of the determinant of the covariance matrices.
        nll_mslogdetcov_expectation, nll_mslogdetcov_variance \
            = self._approximate_nllpslogdetcov(**kwds)

        # Calculate the change in message length.
        current_ll = np.max(predictors.T[2][predictors.T[0] == current_K])
        sign, slogdet = np.linalg.slogdet(self.covariance)

        dI_expectation = dK * (
            (1 - D/2.0)*np.log(2) + 0.25 * (D*(D+3) + 2)*np.log(N/(2*np.pi))) \
            + 0.5 * (D*(D+3)/2.0 - 1) * (slw_expectation - np.sum(np.log(self.weight))) \
            - np.array([np.sum(np.log(current_K + np.arange(_))) for _ in dK])\
            + 0.5 * np.log(_total_parameters(K, D)/float(_total_parameters(current_K, D))) \
            - (D + 2)/2.0 * (np.sum(slogdet)) \
            + current_ll + nll_mslogdetcov_expectation
        
        dI_scatter = nll_mslogdetcov_variance**0.5

        dI_lower_bound = dK * (
            (1 - D/2.0)*np.log(2) + 0.25 * (D*(D+3) + 2)*np.log(N/(2*np.pi))) \
            + 0.5 * (D*(D+3)/2.0 - 1) * (slw_upper - np.sum(np.log(self.weight))) \
            - np.array([np.sum(np.log(current_K + np.arange(_))) for _ in dK])\
            + 0.5 * np.log(_total_parameters(K, D)/float(_total_parameters(current_K, D))) \
            - (D + 2)/2.0 * (np.sum(slogdet)) \
            + current_ll + nll_mslogdetcov_expectation \
            - lower_bound_sigma * dI_scatter

        result = (dI_expectation, dI_scatter, dI_lower_bound)
        return result if np.array(dK).size > 1 \
                      else tuple([_[0] for _ in result])


    def _approximate_sum_log_weights(self, target_K, predictors=None):
        r"""
        Return an approximate expectation of the function:

        .. math:

            \sum_{k=1}^{K}\log{w_k}

        Where :math:`K` is the number of mixtures, and :math:`w` is a multinomial
        distribution. The approximating function is:

        .. math:

            \sum_{k=1}^{K}\log{w_k} \approx -K\log{K}

        :param target_K:
            The number of target Gaussian mixtures.
        """

        if predictors is None:
            predictors = np.array(self._mixture_predictors)

        k, slw = (predictors.T[0], predictors.T[1])

        # Upper bound.
        upper_bound = lambda k, c=0: -k * np.log(k) + c

        #upper = -target_K * np.log(target_K)

        # Some expectation value.
        if 2 > len(k):
            # Don't provide an expectation value.
            expectation = upper_bound(target_K)
            variance \
                = np.abs(upper_bound(target_K**2) - upper_bound(target_K)**2)

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
            expectation = function(target_K, *exp_params)

            #exp_params, exp_cov = op.curve_fit(function, k, slw, p0=[1, 0])
            #expectation = function(target_K, *exp_params)

            variance = 0.0

        return (expectation, variance, upper_bound(target_K))


    def _approximate_nllpslogdetcov(self, target_K, predictors=None,
        draws=100):
        r"""
        Approximate the function:

        .. math:

            -\sum_{n=1}^{N}\log\sum_{k=1}^{K+\Delta{}K}w_{k}f_{k}(y_{n}|\mu_k,C_k) + \frac{(D + 2)}{2}\sum_{k=1}^{(K + \Delta{}K)}\log{|C_k|^{(K+\Delta{}K)}}
        """

        if predictors is None:
            predictors = np.array(self._mixture_predictors)

        k, y = (predictors.T[0], predictors.T[-1])

        k = np.unique(predictors.T[0])
        y = np.empty(k.shape)
        yerr = np.empty(k.shape)

        for i, k_ in enumerate(k):
            match = (predictors.T[0] == k_)
            values = np.log(predictors.T[-1][match])
            y[i] = np.median(values)
            yerr[i] = np.std(values)

        # The zero-th entry of yerr occurs when K = 2, and we only have one
        # estimate of y, so the std is zero.
        #yerr[0] = yerr[1]
        yerr[yerr==0] = np.max(yerr)

        f = lambda x, *p: np.polyval(p, x)
        p0 = np.zeros(2)
        #p0 = np.array([-1, y[0]])
        
        #f = lambda x, *p: np.polyval(p, 1.0/x)
        #p0 = np.hstack([1, np.zeros(min(k.size - 2, 3))])

        op_params, op_cov = op.curve_fit(f, k, y, 
            p0=p0, sigma=yerr, absolute_sigma=True)
        """
        if target_K[0] >= 16:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()

            ax.scatter(k, y)
            ax.fill_between(k, y - yerr, y + yerr, alpha=0.5, zorder=-1)

            op_params, op_cov = op.curve_fit(
            f, k, y, p0=p0, sigma=yerr, absolute_sigma=True)

            ax.plot(k, f(k, *op_params))

            for i in range(4, k.size + 1):

                op_params, op_cov = op.curve_fit(
                f, k[:i], y[:i], p0=p0, sigma=yerr[:i], absolute_sigma=True)
                ax.plot(k[i:] + 1, [f(_, *op_params) for _ in k[i:] + 1], c="g")
                v = np.array([f(_, *op_params) for _ in k[i:] + 1])

                stds = np.array([np.std(f(_, *(np.random.multivariate_normal(op_params, op_cov, size=100).T))) for _ in k[i:] + 1])
                assert np.all(np.isfinite(stds))
                ax.fill_between(k[i:] + 1, v - stds, v + stds, facecolor="g",
                    alpha=0.5)


            
            #log_y = np.empty(k.shape)
            #log_yerr = np.empty(k.shape)

            #for i, k_ in enumerate(k):
            #    match = (predictors.T[0] == k_)
            #    values = np.log(predictors.T[-1][match])
            #    y[i] = np.median(values)
            #    yerr[i] = np.std(values)

            #fig, ax = plt.subplots()
            #ax.scatter(k, y)
            #ax.scatter(k, y + yerr, facecolor="g")



            raise a
        """

        exp_f = lambda x, *p: np.exp(f(x, *p))

        target_K = np.atleast_1d(target_K)
        expectation = np.array([exp_f(tk, *op_params) for tk in target_K])
        
        if not np.all(np.isfinite(op_cov)):
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

            x = k
            ax.scatter(x, y)
            ax.scatter(x, y + yerr, facecolor="g")
            ax.plot(x, f(x, *op_params), c='r')

            fig, ax = plt.subplots()
            ax.scatter(x, np.exp(y))
            ax.scatter(x, np.exp(y + yerr), facecolor='g')

            ax.plot(target_K, expectation, c='r')
            ax.plot(x, exp_f(x, *op_params), c='m')
            variance = np.array([np.var(exp_f(tk, 
                *(np.random.multivariate_normal(op_params, op_cov, size=draws).T)))
                for tk in target_K])
            ax.fill_between(target_K, expectation - variance**0.5, expectation + variance**0.5, facecolor='r', alpha=0.5)

            raise a

        variance = np.array([np.var(exp_f(tk, 
            *(np.random.multivariate_normal(op_params, op_cov, size=draws).T)))
            for tk in target_K])


        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        x = k
        ax.scatter(x, np.exp(y))
        ax.scatter(x, np.exp(y + yerr), facecolor='g')

        ax.plot(target_K, expectation, c='r')
        ax.fill_between(target_K, expectation - variance**0.5, expectation + variance**0.5, facecolor='r', alpha=0.5)
        raise a
        """

        return (expectation, variance)


    def _ftl_jump(self, y, K, **kwargs):
        r"""
        Jump to a totally new mixture of K number of gaussians.
        """

        logger.debug("Re-initializing with K-means++ at K = {}".format(K))

        # Initialize new centroids by k-means++
        mean = kmeans._k_init(y, K, kmeans.row_norms(y, squared=True),
            kmeans.check_random_state(None))

        # Calculate weights by L2 distances to closest centers.
        distance = np.sum((y[:, :, None] - mean.T)**2, axis=1).T

        N, D = y.shape
        responsibility = np.zeros((K, N))
        responsibility[np.argmin(distance, axis=0), np.arange(N)] = 1.0

        weight = responsibility.sum(axis=1)/N

        covariance = _estimate_covariance_matrix(y, responsibility, mean,
            self.covariance_type, self.covariance_regularization)

        mixture = self.__class__(
            threshold=self.threshold,
            covariance_type=self.covariance_type,
            max_em_iterations=self.max_em_iterations,
            covariance_regularization=self.covariance_regularization)

        # Initialize it.
        mixture.set_parameters(mean=mean, weight=weight, covariance=covariance)

        # Run E-M on the partial mixture.
        R, meta = mixture._expectation_maximization(
            y, parent_responsibility=responsibility)

        slogdet = np.sum(np.log(np.linalg.det(mixture.covariance)))
        self._proposed_mixtures.append(mixture)
        self._mixture_predictors.append([
            mixture.weight.size,
            np.sum(np.log(mixture.weight)),
            meta["log_likelihood"],
            slogdet,
            -meta["log_likelihood"] + (D+2)/2.0 * slogdet
        ])
        # TODO: Remove predictors that  we don't use.
        #self._slogs.append(np.linalg.det(mixture.covariance))

        return mixture, R, meta #(len(self._proposed_mixtures) - 1, R, meta)


        raise a

        #self.set_parameters(
        #    weight=weight, mean=mean, covariance=covariance)

        #return responsibility




    def _merge_component_with_closest_component(self, y, responsibility, index, **kwargs):

        

        R, meta, mixture = _merge_components(y, self.mean, self.covariance, self.weight,
            responsibility, index, index_b, **kwargs)

        return mixture, R, meta


    def _optimize_merge_mixture(self, y, responsibility, a_index):

        b_index = _index_of_most_similar_component(y, 
            self.mean, self.covariance, a_index)


        # Initialize.
        weight_k = np.sum(self.weight[[a_index, b_index]])
        responsibility_k = np.sum(responsibility[[a_index, b_index]], axis=0)
        effective_membership_k = np.sum(responsibility_k)

        mean_k = np.sum(responsibility_k * y.T, axis=1) / effective_membership_k
        covariance_k = _estimate_covariance_matrix(
            y, np.atleast_2d(responsibility_k), np.atleast_2d(mean_k), 
            self.covariance_type, self.covariance_regularization)

        # Delete the b-th component.
        del_index = np.max([a_index, b_index])
        keep_index = np.min([a_index, b_index])

        new_mean = np.delete(self.mean, del_index, axis=0)
        new_covariance = np.delete(self.covariance, del_index, axis=0)
        new_weight = np.delete(self.weight, del_index, axis=0)
        new_responsibility = np.delete(responsibility, del_index, axis=0)

        new_mean[keep_index] = mean_k
        new_covariance[keep_index] = covariance_k
        new_weight[keep_index] = weight_k
        new_responsibility[keep_index] = responsibility_k


        mixture = self.__class__(
            threshold=1e-3, # MAGICself.threshold,
            covariance_type=self.covariance_type,
            max_em_iterations=self.max_em_iterations,
            covariance_regularization=self.covariance_regularization)

        mixture.set_parameters(mean=new_mean, weight=new_weight,
            covariance=new_covariance)

        R, meta = mixture._expectation_maximization(
                y, responsibility=new_responsibility)

        #R, ll, I = mixture._expectation(y)
        #meta = {"log_likelihood": ll.sum(), "message_length": I}


        N, D = y.shape
        # Store the mixture.
        slogdet = np.sum(np.log(np.linalg.det(mixture.covariance)))
        self._proposed_mixtures.append(mixture)
        self._mixture_predictors.append([
            mixture.weight.size,
            np.sum(np.log(mixture.weight)),
            meta["log_likelihood"],
            slogdet,
            -meta["log_likelihood"] + (D+2)/2.0 * slogdet
        ])
        # TODO: Remove predictors that  we don't use.
        #self._slogs.append(np.linalg.det(mixture.covariance))

        return (len(self._proposed_mixtures) - 1, R, meta)

        raise a



    def search(self, y, **kwargs):
        r"""
        Simultaneously perform model selection and parameter estimation for an
        unknown number of multivariate Gaussian distributions.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        # Initialize.

        # --> Start on "splitting_mode"

        # --> If we hyperjump, then try merging mode.


        N, D = y.shape

        # Initialize the mixture.
        self._initialize_parameters(y, **kwargs)
        R, ll, I = self._expectation(y, **kwargs)

        converged, just_jumped = (False, False)

        while not converged:

            while True:

                K = self.weight.size
                logger.debug("State: {} {}".format(K, I))

                if just_jumped:

                    
                    # Try to merge components.
                    best_merge = []
                    for k in range(K):
                        try:
                            idx, _, meta = self._optimize_merge_mixture(y, R, k)
                        except:
                            continue

                        logger.debug("Merging: {} {} {}".format(K, k, meta))

                        if k == 0 \
                        or best_merge[-1] > meta["message_length"]:
                            best_merge = [idx, meta["message_length"]]

                        # TODO: Run E-M each time?

                    if best_merge[-1] < I:

                        idx, I = best_merge
                        mixture = self._proposed_mixtures[idx]

                        self.set_parameters(**mixture.parameters)

                        R, ll, I = self._expectation(y, **kwargs)

                        # TODO: Consider hyperjump?
                        continue

                    else:
                        just_jumped = False

                else:
                    # Split all components.
                    best_split = []
                    for k in range(K):
                        idx, _, meta = self._optimize_split_mixture(y, R, k)

                        logger.debug("Splitting: {} {} {}".format(K, k, meta))

                        if k == 0 \
                        or best_split[-1] > meta["message_length"]:
                            best_split = [idx, meta["message_length"]]


                    if best_split[-1] < I:
                        idx, I = best_split
                        mixture = self._proposed_mixtures[idx]

                        self.set_parameters(**mixture.parameters)

                        R, ll, I = self._expectation(y, **kwargs)

                    else:
                        converged = True
                        break

                    # Consider hyperjump.
                    if self.weight.size > 2:

                        K = self.weight.size
                        K_dK = K + np.arange(1, self._predict_mixtures)

                        dI, pI_scatter, dI_lower \
                            = self._predict_message_length_change(K_dK, N)
                        pI = I + dI

                        logger.debug("Actual: {}".format(I))
                        logger.debug("Prediction for next mixture: {}".format(I + dI[0]))
                        logger.debug("Predicted lower bound for next mixture: {}".format(I + dI_lower[0]))
                        logger.debug("Predicted delta for next mixture: {} {}".format(dI[0], pI_scatter[0]))
                        logger.debug("K = {}".format(self.weight.size))


                        ommp = 1 - self._mixture_probability
                        acceptable_jump \
                            = (abs(100 * pI_scatter/pI) < self._percent_scatter) \
                            * (stats.norm(dI, pI_scatter).cdf(0) > self._mixture_probability) 

                        #= (stats.norm(pI, pI_scatter).cdf(I) < ommp) \
                        
                        if dI[0]/pI_scatter[0] < -10 \
                        and not any(acceptable_jump):
                            raise a

                        if any(acceptable_jump):
                        
                            K_jump = K_dK[np.where(acceptable_jump)[0]]
                            # If the jumps are noisy, be conservative.
                            idx = np.where(np.diff(K_jump) > 1)[0]
                            idx = idx[0] if idx else -1

                            K_jump = K_jump[idx]

                            if K_jump - K > 1:

                                logger.debug(
                                    "We should JUMP to K = {}!".format(K_jump))
                                mixture, _, meta = self._ftl_jump(y, K_jump)
                                logger.debug("New meta: {}".format(meta))

                                if meta["message_length"] < I:

                                    # Set the current mixture.
                                    self.set_parameters(**mixture.parameters)
                                    R, ll, I = self._expectation(y, **kwargs)
                                    just_jumped = True


                                else:
                                    #This is a bad jump, so don't accept it.
                                    None

                                

                    # I think we are converged.
                    elif best_split[-1] > I:
                        converged = True
                        break

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        K = self.weight.size
        K_dK = K + np.arange(1, self._predict_mixtures)

        dI, pI_scatter, dI_lower \
                            = self._predict_message_length_change(K_dK, N)
        pI = I + dI     

        ax.scatter(K_dK, pI)
        ax.scatter(K_dK, pI + dI_lower, facecolor="r")

        raise a

        fig, axes = plt.subplots(2)
        axes[0].scatter(y.T[0], y.T[1])
        axes[1].scatter(y.T[0], y.T[2])


        raise a

        """
                # Delete all components.
                K = self.weight.size
                best_merge = []
                if K > 2:
                    # TODO: Some heuristic just to say only try merge if we
                    # hyperjumped?

                    for k in range(K):
                        idx, _, meta = self._optimize_merge_mixture(y, R, k)

                        print("k", k, meta)

                        if k == 0 \
                        or best_merge[-1] > meta["message_length"]:
                            best_merge = [idx, meta["message_length"]]

                        
                # Split all components, and run partial E-M on each.
                K = self.weight.size
                best_perturbation = []

                hyperjump = False
                for k in range(K):
                    # Split the mixture, run partial E-M then full E-M.
                    idx, _, meta = self._optimize_split_mixture(y, R, k)

                    logger.debug(
                        "partial EM {} {} {} {}".format(K, k, idx, meta))

                    # FTL jump!
                    if k > 0 and self.weight.size > 2:

                        K = self.weight.size
                        K_dK = K + np.arange(1, self._predict_mixtures)

                        dI, pI_scatter, dI_lower \
                            = self._predict_message_length_change(K_dK, N)
                        pI = I + dI

                        logger.debug("Actual: {}".format(I))
                        logger.debug("Prediction for next mixture: {}".format(I + dI[0]))
                        logger.debug("Predicted lower bound for next mixture: {}".format(I + dI_lower[0]))
                        logger.debug("Predicted delta for next mixture: {} {}".format(dI[0], pI_scatter[0]))
                        logger.debug("K = {}".format(self.weight.size))


                        ommp = 1 - self._mixture_probability
                        acceptable_jump \
                            = (abs(100 * pI_scatter/pI) < self._percent_scatter) \
                            * (stats.norm(dI, pI_scatter).cdf(0) < ommp) 

                        #= (stats.norm(pI, pI_scatter).cdf(I) < ommp) \
                        
                        if any(acceptable_jump):
                        

                            K_jump = K_dK[np.where(acceptable_jump)[0]]
                            # If the jumps are noisy, be conservative.
                            idx = np.where(np.diff(K_jump) > 1)[0]
                            idx = idx[0] if idx else -1

                            K_jump = K_jump[idx]

                            raise a


                            if K_jump - K > 1:

                                logger.debug(
                                    "We should JUMP to K = {}!".format(K_jump))
                                mixture, R, meta = self._ftl_jump(y, K_jump)

                                logger.debug("New meta: {}".format(meta))

                                # Set the current mixture.
                                self.set_parameters(**mixture.parameters)
                                R, ll, I = self._expectation(y, **kwargs)


                                hyperjump = True
                                break


                    if k == 0 \
                    or best_perturbation[-1] > meta["message_length"]:
                        best_perturbation = [idx, meta["message_length"]]

                if hyperjump:
                    print("Hyperjump EARLY!")
                    continue

                # Is the best perturbation better than the current mixture?
                if best_perturbation[-1] < I and (len(best_merge) == 0 or best_perturbation[-1] < best_merge[-1]):

                    idx, I = best_perturbation
                    mixture = self._proposed_mixtures[idx]

                    self.set_parameters(**mixture.parameters)

                elif len(best_merge) > 0 and best_merge[-1] < I and best_merge[-1] < best_perturbation[-1]:

                    idx, I = best_merge
                    mixture = self._proposed_mixtures[idx]
                    self.set_parameters(**mixture.parameters)
                    
                else:
                    # All split perturbations had longer message lengths.
                    converged = True
                    logger.debug(
                        "All split perturbations had longer message lengths.")
                    break

                # To update message length, max log likelihood tec
                # TODO refactor
                R, ll, I = self._expectation(y, **kwargs)

                # Only start making predictions when we have some data.
                if self.weight.size > 2:

                    K = self.weight.size
                    K_dK = K + np.arange(1, self._predict_mixtures)

                    dI, pI_scatter, dI_lower \
                        = self._predict_message_length_change(K_dK, N)
                    pI = I + dI

                    logger.debug("Actual: {}".format(I))
                    logger.debug("Prediction for next mixture: {}".format(I + dI[0]))
                    logger.debug("Predicted lower bound for next mixture: {}".format(I + dI_lower[0]))
                    logger.debug("Predicted delta for next mixture: {} {}".format(dI[0], pI_scatter[0]))
                    logger.debug("K = {}".format(self.weight.size))


                    ommp = 1 - self._mixture_probability
                    acceptable_jump \
                        = (abs(100 * pI_scatter/pI) < self._percent_scatter) \
                        * (stats.norm(dI, pI_scatter).cdf(0) < ommp) 

                    #= (stats.norm(pI, pI_scatter).cdf(I) < ommp) \
                    
                    if any(acceptable_jump):
                    
                        K_jump = K_dK[np.where(acceptable_jump)[0]]
                        # If the jumps are noisy, be conservative.
                        idx = np.where(np.diff(K_jump) > 1)[0]
                        idx = idx[0] if idx else -1

                        K_jump = K_jump[idx]

                        if K_jump - K > 1:

                            logger.debug(
                                "We should JUMP to K = {}!".format(K_jump))
                            mixture, R, meta = self._ftl_jump(y, K_jump)

                            logger.debug("New meta: {}".format(meta))

                            # Set the current mixture.
                            self.set_parameters(**mixture.parameters)
                            R, ll, I = self._expectation(y, **kwargs)

                    else:
                        # Just split to K+1
                        continue

            if converged:
                logger.debug("Skipping final sweep")
                break

            logger.debug("Doing final sweep")

            # Do a final sweep to be sure.
            K = self.weight.size
            best_perturbation = []
            for k in range(K):
                perturbation = self._propose_split_mixtures(y, R, k)
                if k == 0 \
                or best_perturbation[-1] > perturbation[-1]:
                    best_perturbation = [k] + list(perturbation)

            logger.debug("Actual: {}".format(best_perturbation[-1]))
                
            if best_perturbation[-1] < I:
                k, params, _R, _meta, I = best_perturbation
                self.set_parameters(**params)

                R, ll, I = self._expectation(y, **kwargs)

                # Make a prediction for the next one either way.
                pdf = self._predict_message_length_change(K + 1, N)
                logger.debug("Prediction for next mixture: {}".format(pdf))

            else:
                # Converged.
                converged = True
        """


        logger.debug("Ended at K = {}".format(self.weight.size))
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        fig, ax = plt.subplots()
        ax.scatter(y.T[0], y.T[1], facecolor="#666666", alpha=0.5)

        K = self.weight.size
        for k in range(K):
            mean = self.mean[k][:2]
            cov = self.covariance[k]

            vals, vecs = np.linalg.eigh(cov[:2, :2])
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:,order]

            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

            width, height = 2 * 1 * np.sqrt(vals)
            ellip = Ellipse(xy=mean, width=width, height=height, angle=theta,
                facecolor="r", alpha=0.5)
            ax.add_artist(ellip)
            ax.scatter([mean[0]], [mean[1]], facecolor="r")

        fig, ax = plt.subplots()
        foo = np.array(self._mixture_predictors)
        ax.scatter(foo.T[0], -foo.T[2] - foo.T[3])

        raise a



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




def log_kappa(D):

    cd = -0.5 * D * np.log(2 * np.pi) + 0.5 * np.log(D * np.pi)
    return -1 + 2 * cd/D



def _message_length(y, mu, cov, weight, responsibility, nll,
    covariance_type, eps=0.10, dofail=False, full_output=False, **kwargs):

    # THIS IS SO BAD

    N, D = y.shape
    M = weight.size

    # I(M) = M\log{2} + constant
    I_m = M # [bits]

    # I(w) = \frac{(M - 1)}{2}\log{N} - \frac{1}{2}\sum_{j=1}^{M}\log{w_j} - (M - 1)!
    I_w = (M - 1) / 2.0 * np.log(N) \
        - 0.5 * np.sum(np.log(weight)) \
        - scipy.special.gammaln(M)

    # TODO: why gammaln(M) ~= log(K-1)! or (K-1)!

    #- np.math.factorial(M - 1) \
    #+ 1
    I_w = I_w/np.log(2) # [bits]


    # \sum_{j=1}^{M}I(\theta_j) = -\sum_{j=1}^{M}\log{h\left(\theta_j\right)}
    #                           + 0.5\sum_{j=1}^{M}\log\det{F\left(\theta_j\right)}

    # \det{F\left(\mu,C\right)} \approx \det{F(\mu)}\dot\det{F(C)}
    # |F(\mu)| = N^{d}|C|^{-1}
    # |F(C)| = N^\frac{d(d + 1)}{2}2^{-d}|C|^{-(d+1)}

    # thus |F(\mu,C)| = N^\frac{d(d+3)}{2}\dot{}2^{-d}\dot{}|C|^{-(d+2)}

    # old:
    """
    log_F_m = 0.5 * D * (D + 3) * np.log(N) \
            - 2 * np.log(D) \
            - (D + 2) * np.nansum(np.log(np.linalg.det(cov)))
    """


    # For multivariate, from Kasaraup:
    """
    log_F_m = 0.5 * D * (D + 3) * np.log(N) # should be Neff? TODO
    log_F_m += -np.sum(log_det_cov)
    log_F_m += -(D * np.log(2) + (D + 1) * np.sum(log_det_cov))    
    """
    if D == 1:
        log_F_m = np.log(2) + (2 * np.log(N)) - 4 * np.log(cov.flatten()[0]**0.5)
        raise UnsureError

    else:
        if covariance_type == "diag":
            cov_ = np.array([_ * np.eye(D) for _ in cov])
        else:
            # full
            cov_ = cov

        log_det_cov = np.log(np.linalg.det(cov_))
    
        log_F_m = 0.5 * D * (D + 3) * np.log(np.sum(responsibility, axis=1)) 
        log_F_m += -log_det_cov
        log_F_m += -(D * np.log(2) + (D + 1) * log_det_cov)

        
    # TODO: No prior on h(theta).. thus -\sum_{j=1}^{M}\log{h\left(\theta_j\right)} = 0

    # TODO: bother about including this? -N * D * np.log(eps)
    

    N_cp = _parameters_per_mixture(D, covariance_type)
    part2 = nll + N_cp/(2*np.log(2))

    AOM = 0.001 # MAGIC
    Il = nll - (D * N * np.log(AOM))
    Il = Il/np.log(2) # [bits]

    """
    if D == 1:log_likelihood

        # R1
        R1 = 10 # MAGIC
        R2 = 2 # MAGIC
        log_prior = D * np.log(R1) # mu
        log_prior += np.log(R2)
        log_prior += np.log(cov.flatten()[0]**0.5)

    
    else:
        R1 = 10
        log_prior = D * np.log(R1) + 0.5 * (D + 1) * log_det_cov
    """
    
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
    
        logger.debug(I_m, I_w, np.sum(I_t), lattice, Il)
        logger.debug(I_t)
        logger.debug(part1, part2)

        raise a

    if full_output:
        return (I, dict(I_m=I_m, I_w=I_w, log_F_m=log_F_m, nll=nll, I_l=Il, I_t=I_t,
            lattice=lattice, part1=part1, part2=part2))

    return I


def _index_of_most_similar_component(y, mean, covariance, index):
    r"""
    Find the index of the most similar component, as judged by K-L divergence.
    """
    K, D = mean.shape
    D_kl = np.inf * np.ones(K)
    for k in range(K):
        if k == index: continue
        D_kl[k] = kullback_leibler_for_multivariate_normals(
            mean[index], covariance[index], mean[index], covariance[index])

    return np.nanargmin(D_kl)


def _merge_component_with_closest_component(y, mean, covariance, weight,
    responsibility, index, **kwargs):

    index_b = _index_of_most_similar_component(y, mean, covariance, index)

    return _merge_components(
        y, mean, covariance, weight, responsibility, index, index_b, **kwargs)


def _merge_components(y, mean, covariance, weight, responsibility, index_a,
    index_b, **kwargs):
    r"""
    Merge a component from the mixture with its "closest" component, as
    judged by the Kullback-Leibler distance.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.
    """

    logger.debug("Merging component {} (of {}) with {}".format(
        a_index, weight.size, b_index))

    # Initialize.
    weight_k = np.sum(weight[[a_index, b_index]])
    responsibility_k = np.sum(responsibility[[a_index, b_index]], axis=0)
    effective_membership_k = np.sum(responsibility_k)

    mean_k = np.sum(responsibility_k * y.T, axis=1) / effective_membership_k
    covariance_k = _estimate_covariance_matrix(
        y, np.atleast_2d(responsibility_k), np.atleast_2d(mean_k), 
        kwargs["covariance_type"], kwargs["covariance_regularization"])

    # Delete the b-th component.
    del_index = np.max([a_index, b_index])
    keep_index = np.min([a_index, b_index])

    new_mean = np.delete(mu, del_index, axis=0)
    new_covariance = np.delete(cov, del_index, axis=0)
    new_weight = np.delete(weight, del_index, axis=0)
    new_responsibility = np.delete(responsibility, del_index, axis=0)

    new_mean[keep_index] = mean_k
    new_covariance[keep_index] = covariance_k
    new_weight[keep_index] = weight_k
    new_responsibility[keep_index] = responsibility_k

    # Calculate log-likelihood.
    # Generate a mixture.
    mixture = GaussianMixture()

    raise a

    #return _expectation_maximization(y, new_mean, new_covariance, new_weight,
    #    responsibility=new_responsibility,  **kwargs)



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




def _svd(covariance, covariance_type):

    if covariance_type == "full":
        return np.linalg.svd(covariance)

    elif covariance_type == "diag":
        return np.linalg.svd(covariance * np.eye(covariance.size))

    else:
        raise ValueError("unknown covariance type")

