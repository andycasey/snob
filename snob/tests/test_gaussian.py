
""" Test Gaussian estimator. """

import numpy as np
import unittest

from .. import gaussian


class TestShortHash(unittest.TestCase):


    def test_gaussian_init(self):

        data = np.random.normal(3, 0.5, size=10)
        estimator = gaussian.GaussianEstimator(data)

        


class TestGaussianEstimator(unittest.TestCase):

    def test_repr(self):

        N = 5
        y = np.random.normal(5.2, 0.4, size=N)
        yerr = np.abs(np.random.normal(0, 0.1, size=N))

        model = gaussian.GaussianEstimator(y=y, yerr=yerr)
        print(model)


    def test_attributes(self):


        N = 5
        y = np.random.normal(5.2, 0.4, size=N)
        yerr = np.abs(np.random.normal(0, 0.1, size=N))

        model = gaussian.GaussianEstimator(y=y, yerr=yerr)


        model.data
        model.quantum
        model.message_length
        model.dimensions
        model.parameter_names

        model.log_prior
        model.log_fisher
        model.log_data
