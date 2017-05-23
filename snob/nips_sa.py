
"""
A simulated annealing

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
        mixture_probability=1e-3, threshold=1e-5, max_em_iterations=10000, 
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
        self._mixture_probability = mixture_probability
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
        self._slogs = []


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
        self._proposed_mixtures.append(mixture)
        self._mixture_predictors.append([
            mixture.weight.size,
            np.sum(np.log(mixture.weight)),
            meta["log_likelihood"],
            np.sum(np.log(np.linalg.det(mixture.covariance)))
        ])
        self._slogs.append(np.linalg.det(mixture.covariance))

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



    def _predict_message_length_change(self, N, K=None):
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
        K = current_K + 1 if K is None else int(K)
        if current_K >= K:
            raise ValueError(
                "the target K mixture must contain more Gaussians than the "\
                "current mixture ({} > {})".format(K, current_K))

        predictors = np.array(self._mixture_predictors)
        kwds = dict(target_K=K, predictors=predictors)

        dK = K - current_K
        ll_expectation, ll_scatter \
            = self._approximate_sum_log_likelihood(**kwds)

        slw_expectation, slw_lower, slw_upper \
            = self._approximate_sum_log_weights(**kwds)

        sldc_expectation, sldc_scatter \
            = self._approximate_sum_log_det_covariances(**kwds)

        # Calculate the change in message length.
        current_ll = np.max(predictors.T[2][predictors.T[0] == current_K])
        sign, slogdet = np.linalg.slogdet(self.covariance)

        delta_I = dK * (
            (1 - D/2.0)*np.log(2) + 0.25 * (D*(D+3) + 2)*np.log(N/(2*np.pi))) \
            + 0.5 * (D*(D+3)/2 - 1) * (slw_expectation - np.sum(np.log(self.weight))) \
            - np.sum([np.log(current_K + dk) for dk in range(dK)]) \
            + 0.5 * np.log(_total_parameters(K, D)/float(_total_parameters(current_K, D))) \
            + (D + 2)/2.0 * (
                sldc_expectation
                - np.sum(slogdet)) \
            - ll_expectation + current_ll

        scatter_I = ((sldc_scatter**2 * (D + 2)/2.0) + ll_scatter**2)**0.5

        lower_delta_I = dK * (
            (1 - D/2.0)*np.log(2) + 0.25 * (D*(D+3) + 2)*np.log(N/(2*np.pi))) \
            + 0.5 * (D*(D+3)/2 - 1) * (slw_upper - np.sum(np.log(self.weight))) \
            - np.sum([np.log(current_K + dk) for dk in range(dK)]) \
            + 0.5 * np.log(_total_parameters(K, D)/float(_total_parameters(current_K, D))) \
            + (D + 2)/2.0 * (
                sldc_expectation
                - np.sum(slogdet)) \
            - ll_expectation + current_ll

        return (delta_I, scatter_I, lower_delta_I)


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
        upper = -target_K * np.log(target_K)

        # Some expectation value.
        if 3 > len(k):
            # Don't provide an expectation value.
            fexpectation = None
            variance = None
            lower = None

        else:
            lower_values = [[k[0], slw[0]]]
            for k_, slw_ in zip(k[1:], slw[1:]):
                if k_ == lower_values[-1][0] and slw_ < lower_values[-1][1]:
                    lower_values[-1][1] = slw_
                elif k_ > lower_values[-1][0]:
                    lower_values.append([k_, slw_])
            lower_values = np.array(lower_values)

            function = lambda x, *p: -x * p[0] * np.log(x) + p[1]

            # Lower bound.
            lower_params, lower_cov = op.curve_fit(
                function, lower_values.T[0], lower_values.T[1], p0=[1, 0])
            lower = function(target_K, *lower_params)

            # Expectation.
            exp_params, exp_cov = op.curve_fit(function, k, slw, p0=[1, 0])
            expectation = function(target_K, *exp_params)

        return (expectation, lower, upper)


    def _approximate_sum_log_likelihood(self, target_K, predictors=None,
        samples=30):

        if predictors is None:
            predictors = np.array(self._mixture_predictors)

        k, ll = (predictors.T[0], predictors.T[2])

        upper_values = [[k[0], ll[0]]]
        for k_, ll_ in zip(k[1:], ll[1:]):
            if k_ == upper_values[-1][0] and ll_ > upper_values[-1][1]:
                upper_values[-1][1] = ll_
            elif k_ > upper_values[-1][0]:
                upper_values.append([k_, ll_])
        upper_values = np.array(upper_values)
        
        function = lambda x, *p: p[0] * np.log(x) + p[1]

        # Return an expectation and a variance.
        x, y = upper_values.T
        op_params, op_cov = op.curve_fit(function, x, y, p0=np.ones(2),
            maxfev=100000)

        expectation = function(target_K, *op_params)

        if np.all(np.isfinite(op_cov)):
            scatter = np.std(function(target_K, 
                *(np.random.multivariate_normal(op_params, op_cov, size=samples).T)))

        else:
            scatter = 0

        return (expectation, scatter)


    def _approximate_sum_log_det_covariances(self, target_K, predictors=None):

        if predictors is None:
            predictors = np.array(self._mixture_predictors)
        
        # Only take estimates from the most recent K.
        x, y = predictors.T[0], predictors.T[3]
        idx = 1 + np.where(np.diff(x) > 0)[0][::-1][0]
        
        K = float(x[idx])

        prediction = []
        for xi, yi in zip(x[idx:], self._slogs[idx:]):
            vals = np.array(yi) * float(K)/target_K
            vals = np.hstack([vals, np.median(vals)])
            prediction.append(np.sum(np.log(vals)))

        expectation, scatter = (np.median(prediction), np.std(prediction))
        # TODO: vectorize for target_K

        return expectation, scatter


    def search(self, y, **kwargs):
        r"""
        Simultaneously perform model selection and parameter estimation for an
        unknown number of multivariate Gaussian distributions.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        N, D = y.shape

        # Initialize the mixture.
        self._initialize_parameters(y, **kwargs)
        R, ll, I = self._expectation(y, **kwargs)

        converged = False

        while not converged:

            while True:

                # Split all components, and run partial E-M on each.
                K = self.weight.size
                best_perturbation = []
                incumbent_perturbation = []
                
                for k in range(K):
                    # Split the mixture, run partial E-M then full E-M.
                    idx, _, meta = self._optimize_split_mixture(y, R, k)

                    logger.debug("partial EM {} {} {} {}".format(K, k, idx, meta))
                    if k == 0 \
                    or best_perturbation[-1] > meta["message_length"]:
                        best_perturbation = [idx, meta["message_length"]]

                # Is the best perturbation better than the current mixture?
                if best_perturbation[-1] < I:
                    idx, I = best_perturbation
                    mixture = self._proposed_mixtures[idx]

                    self.set_parameters(**mixture.parameters)
                    # Take this mixture, and calculate a pdf for the next mixture.
                    
                else:
                    # All split perturbations had longer message lengths.
                    converged = True
                    break

                # To update message length, max log likelihood tec
                # TODO refactor
                R, ll, I = self._expectation(y, **kwargs)

                # Only start making predictions when we have a few clusters.
                # TODO: Make this more flexible so that we go from first time.
                if self.weight.size > 2:

                    change, scatter, lower = self._predict_message_length_change(N)
                    
                    logger.debug("Actual: {}".format(I))
                    logger.debug("Prediction for next mixture: {}".format(I + change))
                    logger.debug("Predicted lower bound for next mixture: {}".format(I + lower))
                    logger.debug("Predicted delta for next mixture: {} {}".format(change, scatter))

                    logger.debug("K = {}".format(self.weight.size))

                    if stats.norm(change, scatter).cdf(0) > self._mixture_probability:
                        logger.debug("WE SHOULD SPLIT")

                    else:
                        break

            if converged:
                logger.debug("Skipping final sweep")
                break

            logger.debug("Doing final sweep")

            # Do a final sweep to be sure.
            K = self.weight.size
            best_perturbation = []
            for k in range(K):
                perturbation = self._propose_split_mixture(y, R, k)
                if k == 0 \
                or best_perturbation[-1] > perturbation[-1]:
                    best_perturbation = [k] + list(perturbation)

            logger.debug("Actual: {}".format(best_perturbation[-1]))
                
            if best_perturbation[-1] < I:
                k, params, _R, _meta, I = best_perturbation
                self.set_parameters(**params)

                R, ll, I = self._expectation(y, **kwargs)

                # Make a prediction for the next one either way.
                pdf = self._predict_message_length_change(N)
                logger.debug("Prediction for next mixture: {}".format(pdf))

            else:
                # Converged.
                converged = True


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

