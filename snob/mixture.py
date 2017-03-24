
""" An estimator for a mixture of Gaussians, using minimum message length. """

__all__ = ["GaussianMixtureEstimator"]

import logging
import numpy as np
import scipy.optimize as op

logger = logging.getLogger(__name__)

from . import estimator


class GaussianMixtureEstimator(estimator.Estimator):

    """
    An estimator to model data from (potentially) multiple Gaussian distributions.
    """
    pass