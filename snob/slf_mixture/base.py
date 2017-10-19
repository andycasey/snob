
"""
A base estimator to model data using a fixed number of multivariate Gaussian 
distributions with a single multivariate latent factor.
"""

__all__ = ["BaseMixtureModel"]

import logging
import numpy as np
import scipy

logger = logging.getLogger("snob")


class BaseMixtureModel(object):

    r"""
    Model data from a fixed number of multivariate Gaussian distributions with
    a single (multivariate) latent factor.

    The latent factor and mixture parameters are iteratively updated using
    expectation-maximization, with minimum message length describing the cost
    function.

    :param num_components:
        The number of multivariate Gaussian mixtures to model.

    :param threshold: [optional]
        The relative improvement in log probability required before stopping
        an expectation-maximization step (default: ``1e-5``).

    :param max_em_iterations: [optional]
        The maximum number of iterations to run per expectation-maximization
        loop (default: ``10000``).

    :param initialization_method: [optional]
        The method to use to initialize the mixture parameters. Available
        options are: ``kmeans``, and ``random`` (default: ``kmeans``).
    """

    parameter_names = ("cluster_factor_scores", "factor_scores",
        "factor_loads", "specific_variances", "means", "weights")

    def __init__(self, num_components, threshold=1e-8, max_em_iterations=10000,
        initialization_method="kmeans++", **kwargs):

        num_components = int(num_components)
        if 1 > num_components:
            raise ValueError("number of mixtures must be a positive integer")

        if 0 >= threshold:
            raise ValueError("threshold must be a positive value")

        if 0 >= max_em_iterations:
            raise ValueError("max_em_iterations must be a positive integer")

        available = ("kmeans++", "random")
        initialization_method = initialization_method.strip().lower()
        if initialization_method not in available:
            raise ValueError(
                "Initialization method '{}' is invalid. Must be one of: {}"\
                .format(initialization_method, ", ".join(available)))

        self.num_components = num_components
        self.threshold = threshold
        self.max_em_iterations = max_em_iterations
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
        Return the factor scores, :math:`a_k`.
        """
        return self._factor_scores


    @property
    def cluster_factor_scores(self):
        r"""
        Return the mean factor scores for each :math:`k`-th cluster,
        :math:`v_k`.
        """
        return self._cluster_factor_scores


    @property
    def cluster_factor_score_variances(self):
        r"""
        Return the variance in the factor scores for each :math:`k`-th cluster.
        """
        return self._cluster_factor_score_variances


    @property
    def specific_variances(self):
        r"""
        Return the specific variances, :math:`\sigma_k^2`.
        """
        return self._specific_variances


    def set_parameters(self, **kwargs):
        r"""
        Set specific parameters.
        """

        for parameter_name, value in kwargs.items():
            if parameter_name in self.parameter_names:
                pn = "_{}".format(parameter_name)
                value = value if value is None else np.atleast_2d(value)
                
                existing_value = getattr(self, pn, None)
                if hasattr(self, "truth"):
                    try:
                        true_value = self.truth[parameter_name]
                    except:
                        if parameter_name == "cluster_factor_scores":
                            true_value = np.unique(self.truth["factor_scores"])
                        else:
                            raise
                else:
                    true_value = None
                        
                setattr(self, pn, value)

                
                if true_value is not None and existing_value is not None:
                    diff_before = np.sum(np.abs(existing_value - value))
                    diff_now = np.sum(np.abs(true_value - value))

                    logger.debug("set_parameters: {} {} (before: {}; truth {}) diffs (before: {}; now: {}) {}".format(
                        parameter_name, value, existing_value, true_value, diff_before, diff_now, "GOOD" if diff_now < diff_before else "BAD"))


                else:


                    if parameter_name == "specific_variances" and value is not None:
                        logger.debug("set_parameters: {} {} (specific_sigmas from {} to {})".format(
                            parameter_name, value, np.sqrt(existing_value) if existing_value is not None else None, np.sqrt(value)))
                    else:
                        logger.debug("set_parameters: {} from {} to {}".format(
                            parameter_name, existing_value, value))

            else:
                raise ValueError("unknown parameter '{}'".format(parameter_name))

        return True


    def _aecm_step_1(self, *args, **kwargs):
        raise NotImplementedError("this must be implemented by sub-classes")


    def _aecm_step_2(self, *args, **kwargs):
        raise NotImplementedError("this must be implemented by sub-classes")


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

        self.initialize_parameters(data)

        # Calculate the inital expectation.
        responsibility, log_likelihood = self._expectation(data) 

        results = [log_likelihood.sum()]
        I = [self.message_length(data)]

        logger.debug("fit: initial ll/I: {} {}".format(results[0], I[0]))

        for iteration in range(self.max_em_iterations):

            # Do conditional updates.
            # AECM
            # Do the CM-step 1, where we update the weights and means.
            self._aecm_step_1(data, responsibility)

            # Compute the e-step of cycle 2
            responsibility, _ = self._expectation(data)
            logger.debug("fit intermediate ll: {}".format(_.sum()))

            # Do the M-step of cycle 2
            self._aecm_step_2(data, responsibility)

            # Re-compute the expectation,
            responsibility, log_likelihood_ = self._expectation(data)
            #assert log_likelihood_.sum() > log_likelihood.sum()
            log_likelihood = log_likelihood_
            results.append(log_likelihood.sum())
            I.append(self.message_length(data))


            # Check for convergence.
            metric = I
            change = np.abs((metric[-1] - metric[-2])/metric[-2])
            
            logger.debug("fit: {} {} {} {}".format(
                iteration, results[-1], change, I[-1]))

            if not np.isfinite(I[-1]) or not np.isfinite(results[-1]):
                raise wtf
            
            if change <= self.threshold \
            or iteration >= self.max_em_iterations:
                break

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.scatter(data.T[0], data.T[1], c="#666666", alpha=0.5)

        N, D = data.shape
        colors = "br"
        for i in range(self.num_components):
            foo = self.means + np.dot(self.cluster_factor_scores.flatten()[i], self.factor_loads)
            ax.scatter(foo.T[0], foo.T[1], facecolor=colors[i])

        fig, ax = plt.subplots()

        transformed = data - self.means - np.dot(self.approximate_factor_scores, self.factor_loads)
        ax.scatter(transformed.T[0], transformed.T[1], c=responsibility[0])

        return self


    def _expectation(self, data, full_output=False):
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
        
        # Store the responsibility matrix because it is helpful to use later on
        self._responsibility = responsibility

        return (responsibility, log_likelihood, weighted_log_prob) \
            if full_output else (responsibility, log_likelihood)


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
            data, self.factor_loads, self.approximate_factor_scores, 
            self.specific_variances, self.means)



def _estimate_log_latent_factor_prob(data, factor_loads, cluster_factor_scores,
    specific_variances, mean):

    N, D = data.shape
    L, K = cluster_factor_scores.shape

    # As usual:
    # N = number of data points
    # D = dimensions of data
    # L = number of latent factors
    # K = number of clusters.

    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots()
    #ax.scatter(data.T[0], data.T[1], facecolor="#666666", zorder=-1)
    #colors = "br"

    # TODO consider multiplication of inv_specific_variances instead
    #import matplotlib.pyplot as plt
    #fig, ax = plt.subplots()
    #ax.scatter(data.T[0], data.T[1], facecolor="#666666", zorder=-1, alpha=0.5)
    #colors = "br"

    log_prob = np.empty((N, K))
    #for k, factor_scores in enumerate(cluster_factor_scores.T):
    #    squared_diff = (data - mean - np.dot(factor_scores, factor_loads))**2
    #    log_prob[:, k] = np.sum(squared_diff / specific_variances, axis=1)

    squared_diff = (data - mean - np.dot(cluster_factor_scores, factor_loads))**2
    log_prob[:, 0] = np.sum(squared_diff / specific_variances, axis=1)


    #raise a

    return - 0.5 * K * N * np.log(2 * np.pi) \
           + N * np.sum(np.log(specific_variances)) \
           - 0.5 * log_prob

