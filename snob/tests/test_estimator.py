
""" Test base estimator. """

import numpy as np
import unittest

from .. import estimator


class TestBaseEstimator(unittest.TestCase):

    def test_attributes(self):

        model = estimator.Estimator()

        self.assertTrue(model.quantum > 0)
         
        with self.assertRaises(NotImplementedError):
            model.log_prior

        with self.assertRaises(NotImplementedError):
            model.log_fisher

        with self.assertRaises(NotImplementedError):
            model.log_data

        with self.assertRaises(NotImplementedError):
            model.parameter_names


        model._data = [3]
        self.assertIsNotNone(model.data)