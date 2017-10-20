
"""
A single latent factor model.
"""

__all__ = ["SLFModel"]

import logging
import numpy as np
from scipy import optimize as op

logger = logging.getLogger("snob")


class SLFModel(object):

    r"""
    Model data with a single (multivatiate) latent factor.

    :param threshold: [optional]
        The relative improvement in log probability required before stopping
        an expectation-maximization step (default: ``1e-5``).

    :param max_em_iterations: [optional]
        The maximum number of iterations to run per expectation-maximization
        loop (default: ``10000``).
    """

    parameter_names = ("factor_scores", "factor_loads", "specific_variances", 
        "means")

    def __init__(self, threshold=1e-8, max_em_iterations=10000, **kwargs):

        if 0 >= threshold:
            raise ValueError("threshold must be a positive value")

        if 0 >= max_em_iterations:
            raise ValueError("max_em_iterations must be a positive integer")

        self.threshold = threshold
        self.max_em_iterations = max_em_iterations
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

                setattr(self, pn, value)
                #logger.debug("set_parameters: {} {}".format(pn, value))

            else:
                raise ValueError("unknown parameter '{}'".format(parameter_name))

        return True


    def initialize_parameters(self, data, **kwargs):
        r"""
        Initialize the model parameters, given the data.

        :param data:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is
            the number of dimensions per observation.
        """

        # Estimate the latent factor first.
        N, D = data.shape

        # Excuse the terminology problem here, but means here refers to the 
        # single mean for the entire sample, not the cluster means!
        # TODO REVISIT TERMINOLOGY
        self.set_parameters(means=np.mean(data, axis=0))
        
        w = data - self.means
        V = np.dot(w.T, w)
        
        # Set \beta as the dominant eigenvector of the correlation matrix with
        # length given by setting b^2 equal to the dominant eigenvalue.
        _, s, v = np.linalg.svd(np.corrcoef(data.T))
        beta = v[0] * np.sqrt(s[0])

        # Iterate as per Wallace & Freeman first.
        for i in range(self.max_em_iterations):
            beta, change = self._iterate_and_update(data, beta, V)
            logger.debug("initialize_parameters: {} {}".format(i, change))
            if change is None or change < self.threshold: break
        
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

        self.initialize_parameters(data)

        N, D = data.shape

        def pack(*params):
            return np.hstack([np.array(p).flatten() for p in params])

        def unpack(params):
            factor_scores = params[:N].reshape((N, 1))
            factor_loads = params[N:N + D].reshape((1, D))
            specific_sigmas = params[N + D:N + 2*D].reshape((1, D))
            return (factor_scores, factor_loads, specific_sigmas)


        def objective_function(params):
            factor_scores, factor_loads, specific_sigmas = unpack(params)
            return _message_length(data, self.means, factor_scores, 
                factor_loads, specific_sigmas**2)

        p0 = pack(self.factor_scores, self.factor_loads, self.specific_variances)
        
        p_opt, I, meta = op.fmin_l_bfgs_b(objective_function, p0, 
            approx_grad=True, factr=10, maxfun=10**6, maxiter=10**6)

        factor_scores, factor_loads, specific_sigmas = unpack(p_opt)
        self.set_parameters(
            factor_scores=factor_scores,
            factor_loads=factor_loads,
            specific_variances=specific_sigmas**2)
        
        return (p_opt, I, meta)



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

        log_likelihood = self._estimate_log_prob(data)
        responsibility = None
        
        # Store the responsibility matrix because it is helpful to use later on
        self._responsibility = responsibility

        return (responsibility, log_likelihood)





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



    def message_length(self, y, full_output=False):
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

        return _message_length(
            y, self.means, self.factor_scores, self.factor_loads,
            self.specific_variances, full_output=full_output)


    def _iterate_and_update(self, data, beta, V):

        N, D = data.shape

        beta, specific_variances, change = self._iterate_wf90(beta, V, N, D)

        factor_loads = np.sqrt(specific_variances) * beta
        b_sq = np.sum(beta**2)
        #factor_scores = np.dot(w/specific_sigmas, beta) \
        #              * ((1.0 - D/((N - 1.0) * b_sq)) / (1 + b_sq))
        factor_scores = np.dot(data, beta) \
                      * ((1.0 - D/(N - 1.0) * b_sq) / (1 + b_sq))
        factor_scores = np.atleast_2d(factor_scores).T

        self.set_parameters(
            factor_scores=factor_scores,
            factor_loads=factor_loads,
            specific_variances=specific_variances)

        return (beta, change)



    def _iterate_wf90(self, beta, V, N, D):

        beta = beta.reshape((D, ))

        b_squared = np.sum(beta**2)

        lhs = (N - 1.0) * b_squared
        if lhs <= D:
            logger.warn("Setting \Beta_k = 0 as {} <= {}".format(lhs, D))
            beta, b_squared = (np.zeros(D), 0)

        # Compute specific variances according to:
        #   \sigma_k^2 = \frac{\sum_{n}w_{nk}^2}{(N-1)(1 + \Beta_k^2)}
        # (If \Beta = 0, exit)

        # Discrepancy between Wallace (2005) and Wallace and Freeman (1992) here
        # as to whether this should be \beta_k^2 or b^2
        specific_variances = np.diag(V) / ((N - 1.0) * (1.0 + beta**2))

        assert specific_variances.size == D

        if np.all(beta == 0):
            logger.warn("Exiting step (b) of scheme in initialisation")
            return (beta, specific_variances, None)

        # Compute Y using 
        #   Y_{kj} = V_{kj}/\sigma_{k}\sigma_{j}
        # which is Wallace & Freeman nomenclature. In our nomenclature
        # that is:
        #   Y_{ij} = V_{ij}/\sigma_{i}\sigma_{j}

        specific_sigmas = np.atleast_2d(specific_variances**0.5)
        Y = V / np.dot(specific_sigmas.T, specific_sigmas)

        # Compute an updated estimate of \beta as:
        #   \Beta_{new} = \frac{Y\Beta(1-K/(N-1)b^2)}{(N - 1)(1 + b^2)}
        # which is Wallace & Freeman nomenclature. In our nomenclature
        # that is:
        #   \Beta_{new} = \frac{Y\Beta(1-D/(N - 1)b^2)}{(N - 1)(1 + b^2)}

        # Where you will recall (in our nomenclature)
        #   b^2 = \sum_{D} b_d^2
        # And b_{d} = a_{d}/\sigma_{d}

        b_squared = np.sum(beta**2)
        beta_new = (np.dot(Y, beta) * (1.0 - D/(N - 1.0) * b_squared)) \
                 / ((N - 1.0) * (1.0 + b_squared))

        # If \Beta_new \approx \Beta then exit.
        l1_norm = np.abs(beta - beta_new).sum()

        if not np.isfinite(l1_norm):
            raise ValueError("l1_norm in _iterate_wallace_freeman_1990 is not finite")

        return (beta_new, specific_variances, l1_norm)



def _estimate_log_latent_factor_prob(data, factor_loads, factor_scores,
    specific_variances, mean):

    N, D = data.shape

    # As usual:
    # N = number of data points
    # D = dimensions of data
    # L = number of latent factors
    # K = number of clusters.

    squared_diff = (data - mean - np.dot(factor_scores, factor_loads))**2
    log_prob = np.sum(squared_diff / specific_variances)
    
    assert np.isfinite(log_prob).all()

    nll = 0.5 * D * N * np.log(2 * np.pi) \
        + N * np.sum(np.log(np.sqrt(specific_variances))) \
        + 0.5 * log_prob
    return -nll


def _message_length(y, means, factor_scores, factor_loads, specific_variances,
    yerr=0.001, full_output=False):

    N, D = y.shape

    yerr = np.array(yerr)
    if yerr.size == 1:
        yerr = yerr * np.ones_like(y)
    elif yerr.shape != y.shape:
        raise ValueError("shape mismatch")

    N, D = y.shape

    v_sq = np.sum(factor_scores**2)
    b_sq = np.sum(factor_loads**2 / specific_variances)
    S = np.sum(factor_scores)

    residual = y - means - np.dot(factor_scores, factor_loads)

    I = {
        "L1": (N - 1.0) * np.sum(np.log(np.sqrt(specific_variances))),
        "L2": 0.5 * (D * np.log(N * v_sq - S**2) + (N + D - 1) * np.log(1 + b_sq)),
        "L3": 0.5 * (v_sq + np.sum((residual)**2 / specific_variances))
    }

    #I["lattice"] = 0.5 * N_free_params * log_kappa(N_free_params) 
    #I["constant"] = 0.5 * N_free_params

    I_total = np.hstack(I.values()).sum() / np.log(2) # [bits]


    #LL = -_estimate_log_latent_factor_prob(y, factor_loads, factor_scores,
    #    specific_variances, means)
    #print(LL)
    #return LL
    print(I_total, I)
    
    if not full_output:
        return I_total

    for key in I.keys():
        I[key] /= np.log(2) # [bits]

    return (I_total, I)



