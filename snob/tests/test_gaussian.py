
""" Test Gaussian estimator. """

import numpy as np
import unittest

from .. import gaussian


class TestShortHash(unittest.TestCase):


    def test_gaussian_init(self):

        data = np.random.normal(3, 0.5, size=10)
        estimator = gaussian.GaussianEstimator(data)

        