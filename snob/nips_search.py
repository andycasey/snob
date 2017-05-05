
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


def _generate_ll_approximator(K, log_likelihood,
    normalization_factor, *log_likelihoods_of_sequentially_increasing_mixtures):
    
    x = np.arange(K, K + len(log_likelihoods_of_sequentially_increasing_mixtures))
    ll = np.hstack([log_likelihood, *log_likelihoods_of_sequentially_increasing_mixtures])
    y = np.diff(ll) / normalization_factor

    functions = {
        1: lambda x, *p: p[0] / np.exp(x),
        2: lambda x, *p: p[0] / np.exp(p[1] * x),
        3: lambda x, *p: p[0] / np.exp(p[1] * x) + p[2]
    }

    foo = False
    if x.size > 3:
        #x = x[1:]
        #y = y[1:]
        foo = True

    p_opt_ = []
    for d in np.arange(1, 1 + x.size)[::-1]:

        cost_function = functions.get(d, functions[3])
        p0 = np.ones(d)

        try:
            p_opt, p_cov = op.curve_fit(cost_function, x, y, p0=p0)
        except:
            assert d > 1
            continue
        else:
            p_opt_.append(p_opt)
            break

    p_opt = p_opt_.pop(0)
    d = p_opt.size
    cost_function = functions.get(d, functions[3])

    generating_function = lambda target_K: log_likelihood \
        + cost_function(np.arange(1, target_K), *p_opt).sum() * normalization_factor

    if False and foo and x.size > 10:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(np.arange(ll.size) + 1, ll)
        ax.plot(np.arange(ll.size) + 1, [generating_function(_) for _ in np.arange(ll.size) + 1], c='r')
        raise a

    return generating_function



def _approximate_log_likelihood_improvement(y, mu, cov, weight, 
    log_likelihood, *log_likelihoods_of_sequentially_increasing_mixtures):
    """
    Return a function that will approximate the log-likelihood value for a
    given target number of mixtures.
    """

    # Evaluate the current mixture.
    K = weight.size
    evaluate_f1 = np.sum(weight * np.vstack(
        [_evaluate_gaussian(y, mu[k], cov[k]) for k in range(K)]).T)

    x = np.arange(K, K + len(log_likelihoods_of_sequentially_increasing_mixtures))
    ll = np.hstack([log_likelihood, log_likelihoods_of_sequentially_increasing_mixtures])
    y = np.diff(ll) / evaluate_f1

    d = x.size
    functions = {
        1: lambda x, *p: p[0] / np.exp(x),
        2: lambda x, *p: p[0] / np.exp(p[1] * x),
        3: lambda x, *p: p[0] / np.exp(p[1] * x) + p[2]
    }
    cost_function = functions.get(d, functions[3])
    p0 = np.ones(d)

    p_opt, p_cov = op.curve_fit(cost_function, x, y, p0=p0)
    
    # Now generate the function to estimate the log-likelihood of the K-th
    # mixture.
    generating_function = lambda K: log_likelihood + cost_function(np.arange(1, K-1), *p_opt).sum() * evaluate_f1

    return generating_function



def _approximate_sum_log_weights(target_K):
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

    # TODO: Return a variance on this expectation.
    return -target_K * np.log(target_K)


def _approximate_bound_slogdet_covariances(target_K, 
    covariance_matrices, covariance_type):
    r"""
    Return an approximate expectation of the function:

    .. math:

        \sum_{k=1}^{K}\log{|\bm{C}_k|}

    Where :math:`C_k` is the covariance matrix of the :math:`K`-th mixture.
    A lower and upper bound is given for the approximation, which is based on
    the current estimates of the covariance matrices. The determinants of
    the target distribution are estimated as:

    .. math:

        |\bm{C}_k| = \frac{k_{current}}{k_{target}}\min\left(|C_k|\right)
    
    and the upper bound as:

    .. math:

        |\bm{C}_k| = \frac{k_{current}}{k_{target}}\max\left(|C_k|\right)

    :param K:
        The target number of Gaussian mixtures.
    
    :param covariance_matrices:
        The current estimate of the covariance matrices.
    
    :param covariance_type:
        The type of structure assumed for the covariance matrix.

    :returns:
        An estimated lower and upper bound on the sum of the logarithm of the
        determinants of a :math:`K` Gaussian mixture.
    """

    # Get the current determinants.
    current_K, D, _ = covariance_matrices.shape
    assert covariance_type == "full" #TODO
    assert target_K > current_K

    current_det = np.linalg.det(covariance_matrices)

    current_det_bounds = np.array([np.min(current_det), np.max(current_det)])
    target_det_bounds = current_K/float(target_K) * current_det_bounds

    return target_K * np.log(target_det_bounds)


def _approximate_message_length_change(target_K, current_weights,
    current_cov, current_log_likelihood, N, initial_ll, normalization_factor, optimized_mixture_lls,
    current_ml=0):
    r"""
    Estimate the change in message length between the current mixture of 
    Gaussians, and a target mixture.
    """

    func = _generator_for_approximate_log_likelihood_improvement(1, initial_ll,
        normalization_factor, *np.hstack([optimized_mixture_lls, current_log_likelihood]))

    current_K, D, _ = current_cov.shape
    delta_K = target_K - current_K
    assert delta_K > 0

    # Calculate everything except the log likelihood.
    delta_I = delta_K * (
            (1 - D/2.0) * np.log(2) \
            + 0.25 * (D * (D+3) + 2) * np.log(N/(2*np.pi))) \
        + 0.5 * (D*(D+3)/2 - 1) * (_approximate_sum_log_weights(target_K) - np.sum(np.log(current_weights))) \
        - np.sum([np.log(current_K + dk) for dk in range(delta_K)]) \
        + 0.5 * np.log(_total_parameters(target_K, D)/float(_total_parameters(current_K, D))) \
        + (D + 2)/2.0 * (
            _approximate_bound_sum_log_determinate_covariances(target_K, current_cov, "full") \
            - np.sum(np.log(np.linalg.det(current_cov)))) \
        - func(target_K) + current_log_likelihood
    # Generate a function.
    logger.debug("PREDICTING TARGET {} FROM {}: {}".format(target_K, current_weights.size, delta_I))
    if delta_K == 1:
        assert np.all(delta_I + current_ml > 0)

    return delta_I




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
        
        # For predictions.
        self.__slog_weight = []
        self.__slogdet_covariance = []
        self._max_mixture_log_likelihood = {}
        self._min_mixture_message_length = {}

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


    @property
    def mixture_probability(self):
        r""" 
        Return the minimum probability needed before we add a new mixture. 
        """
        return self._mixture_probability


    def _expectation(self, y, **kwargs):
        r"""
        Perform the expectation step of the expectation-maximization algorithm.

        :param y:
            The data values, :math:`y`.

        :returns:
            A three-length tuple containing the responsibility matrix,
            the  log likelihood, and the change in message length.
        """

        responsibility, log_likelihood \
            = self.responsibility_matrix(y, full_output=True, **kwargs)

        ll = np.sum(log_likelihood)

        I = _message_length(y, self.mean, self.covariance, self.weight,
            responsibility, -ll, self.covariance_type,
            **kwargs)

        # Update our record of the best log-likelihood per mixture,
        # so that we can make predictions about future mixtures.
        K = self.weight.size
        assert ll > self._max_mixture_log_likelihood.get(K, -np.inf)
        #assert I < self._min_mixture_message_length.get(K, np.inf)
        self._max_mixture_log_likelihood[K] = ll
        self._min_mixture_message_length[K] = I

        return (responsibility, log_likelihood, I)


    def _maximization(self, y, responsibility, **kwargs):
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
        
        effective_membership = np.sum(responsibility, axis=1)
        weight = (effective_membership + 0.5)/(N + K/2.0)

        mean = np.empty(self.mean.shape)
        for k, (R, Nk) in enumerate(zip(responsibility, effective_membership)):
            mean[k] = np.sum(R * y.T, axis=1) / Nk

        covariance = _estimate_covariance_matrix(y, responsibility, mean, 
            self.covariance_type, self.covariance_regularization)

        # TODO: Callback?
        return self.set_parameters(
            weight=weight, mean=mean, covariance=covariance)


    def _split_component(self, y, responsibility, component_index, **kwargs):
        r"""
        Split a component from the current mixture.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is the
            number of dimensions per observation.

        :param component_index:
            The index of the component to be split.

        """

        # Compute the direction of maximum variance of the parent component, 
        # and locate two points that are one standard deviation on either side.
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

        weight = np.hstack([self.weight, [parent_weight * split_weight[1]]])
        weight[component_index] = parent_weight * split_weight[0]

        mean = np.vstack([self.mean, [split_mean[1]]])
        mean[component_index] = split_mean[0]

        covariance = np.vstack([self.covariance, [split_covariance[1]]])
        covariance[component_index] = split_covariance[0]

        # Set parameters.
        return self.set_parameters(mean=mean, weight=weight, covariance=covariance)


    def responsibility_matrix(self, y, full_output=False, **kwargs):
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

        precision_cholesky = _compute_precision_cholesky(
            self.covariance, self.covariance_type)
        weighted_log_prob = np.log(self.weight) + _estimate_log_gaussian_prob(
            y, self.mean, precision_cholesky, self.covariance_type)

        log_likelihood = scipy.misc.logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            log_responsibility = weighted_log_prob - log_likelihood[:, np.newaxis]

        responsibility = np.exp(log_responsibility).T
        
        return responsibility if not full_output \
                              else (responsibility, log_likelihood)


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

            # Record attributes of these values for predictive purposes.
            # TODO: We could probably save this somewhere else, when it is
            # computed for reasons that we actually need it.
            if value is not None: # And a keyword argument to prevent?
                if parameter_name == "weight":
                    self.__slog_weight.append([value.size, np.sum(np.log(value))])

                elif parameter_name == "covariance":
                    sign, slogdet = np.linalg.slogdet(value)
                    self.__slogdet_covariance.append([value.size, value.sum()])


        return kwargs


    def _predict_mixture_minimum_message_length(self, N, K=None):
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

        # TODO: Normalization factor?
        # TODO: Put generator in the class?
        # TODO: data type for optimized mixture
        lls = [self._max_mixture_log_likelihood[k] for k in sorted(self._max_mixture_log_likelihood.keys())[1:]]
        approximate_ll = _generate_ll_approximator(1, self._max_mixture_log_likelihood[1], 1, *lls)

        dK = K - current_K

        # TODO: Function to improve approximation of sum of log weights.
        # TODO: Function to improve approximate bounds of sum_log_determinate_covariances

        # Calculate the change in message length.
        sign, slogdet = np.linalg.slogdet(self.covariance)
        assert np.all(sign > 0), "Whoa hold on there buddy"

        delta_I = dK * (
            (1 - D/2.0)*np.log(2) + 0.25 * (D*(D+3) + 2)*np.log(N/(2*np.pi))) \
            + 0.5 * (D*(D+3)/2 - 1) * (_approximate_sum_log_weights(K) - np.sum(np.log(self.weight))) \
            - np.sum([np.log(current_K + dk) for dk in range(dK)]) \
            + 0.5 * np.log(_total_parameters(K, D)/float(_total_parameters(current_K, D))) \
            + (D + 2)/2.0 * (
                _approximate_bound_slogdet_covariances(K, self.covariance, self.covariance_type)
                - np.sum(slogdet)) \
            - approximate_ll(K) + lls[-1]

        I = delta_I + self._min_mixture_message_length[current_K]
        return I


    def _approximate_sum_log_weights(self, target_K):
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

        # If we have no information, then the function is:

        return -target_K * np.log(target_K)

        # But if we have information then we should make the function flexible
        # and keep the best estimates.



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

        while True:

            # Split components, in order of ones with highest |C|.
            component_indices = np.argsort(np.linalg.det(self.covariance))[::-1]
            split_component_index = component_indices[0]

            logger.debug("Splitting component {} from {} to {}".format(
                split_component_index,
                self.weight.size, self.weight.size + 1))
            
            # Split this mixture.
            R = self._split_component(y, R, split_component_index)
            
            R, ll, I = self._expectation(y)
            lls = [ll.sum()]

            for iteration in range(self.max_em_iterations):

                # Maximize the parameters, conditioned on our expectation.
                self._maximization(y, R)

                # Calculate P(I_{K+1} - I_{K} < 0).
                # If we should add another component, then run the split
                # again.
                
                # If not, then we should check to see if E-M is converged.
                # If it is, then stop. If not, run E-M again.
                R, ll, I = self._expectation(y)
                lls.append(ll.sum())

                change = (lls[-1] - lls[-2])/lls[-2]
                logger.debug("E-M step: {} {}".format(iteration, change))
                pdf = self._predict_mixture_minimum_message_length(N)
                
                if self.threshold >= abs(change):
                    # Here we want to leave the loop and not allow any more
                    # splitting, since it's probably not worthwhile doing.
                    
                    
                    logger.debug("Prediction from {}: {} {} {} --> {}".format(
                        self.weight.size, iteration, pdf - I, I,
                        "SPLIT" if np.percentile(pdf, 1 - self._mixture_probability) < I else "KEEP"))
                    if np.percentile(pdf, 1 - self._mixture_probability) < I:
                        # Don't do any more E-M iterations, just split a component
                        logger.debug("SPLIT")
                    else:
                        converged = True



                    break


            else:
                logger.debug("Warning: max number of EM steps")


            # Did that E-M iteration converge, and we won't split again?
            if converged: 
                logger.debug("Convergence detected")
                break


            # TODO:
            # Need some kind of check to see whether we should have
            # splitted something else.
            logger.debug("CHECK SOMETHING")

        logger.debug("K = {}".format(self.weight.size))
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        fig, ax = plt.subplots()
        ax.scatter(y.T[0], y.T[1], facecolor="#666666", alpha=0.5)

        K = self.weight.size
        for k in range(K):
            mean = self.mean[k]
            cov = self.covariance[k]

            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[order]

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


