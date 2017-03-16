
""" Test base estimator. """

import numpy as np
import unittest

from .. import estimator


class TestBaseEstimator(unittest.TestCase):

    def test_attributes(self):

        N = 5
        data = np.random.uniform(size=N)
        model = estimator.Estimator(data=y)
        
        model.data
        model.quantum
        model.message_length
    
        with self.assertRaises(NotImplementedError):
            model.log_prior

        with self.assertRaises(NotImplementedError):
            model.log_fisher

        with self.assertRaises(NotImplementedError):
            model.log_data
