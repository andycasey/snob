
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


    def test_negative_error(self):

        with self.assertRaises(ValueError):
            gaussian.GaussianEstimator(y=[0], yerr=[-1])


    def test_data_shape_mismatch(self):

        with self.assertRaises(ValueError):
            gaussian.GaussianEstimator(y=[0], yerr=[0.1, 0.1])

        with self.assertRaises(ValueError):
            gaussian.GaussianEstimator(y=[1, 1], yerr=0.1)


    def test_proper_prior_on_mean(self):

        with self.assertRaises(ValueError):
            gaussian.GaussianEstimator(y=[1], mean_bounds=(None, 3))

        with self.assertRaises(ValueError):
            gaussian.GaussianEstimator(y=[1], mean_bounds=(3, None))

        self.assertIsNotNone(
            gaussian.GaussianEstimator(y=[1], mean_bounds=(None, None)))

        self.assertIsNotNone(
            gaussian.GaussianEstimator(y=[1], mean_bounds=None))

        with self.assertRaises(ValueError):
            gaussian.GaussianEstimator(y=[1], mean_bounds=[3])

        with self.assertRaises(ValueError):
            gaussian.GaussianEstimator(y=[1], mean_bounds=[1,2,3])


        bounds = [5, 1]
        model = gaussian.GaussianEstimator(y=[2], mean_bounds=bounds)
        self.assertEqual(model.bounds[0][0], bounds[1])




    def test_optimization(self):

        N = 5
        y = np.random.normal(5.2, 0.4, size=N)
        yerr = np.abs(np.random.normal(0, 0.1, size=N))

        model = gaussian.GaussianEstimator(y=y, yerr=yerr)

        before = model.message_length

        model.optimize()

        after = model.message_length

        self.assertTrue(after <= before)

        # Ensure we hit the warning flag.
        model.optimize(maxiter=0, maxfun=0, factr=10, pgtol=1e-30)



    def test_prior_on_mean(self):

        N = 5
        y = np.random.normal(5.2, 0.4, size=N)
        yerr = np.abs(np.random.normal(0, 0.1, size=N))

        without_prior = gaussian.GaussianEstimator(y=y, yerr=yerr)
        without_prior.optimize()

        with_prior = gaussian.GaussianEstimator(
            y=y, yerr=yerr, mean_bounds=[1, 10])
        with_prior.optimize()

        print("with", with_prior.message_length)
        print("without", without_prior.message_length)
        #self.assertTrue(
        #    with_prior.message_length > without_prior.message_length)

        print("TODO CHECK PRIOR ON MEAN")


    def test_prior_on_sigma(self):




        N = 5
        y = np.random.normal(5.2, 0.4, size=N)
        yerr = np.abs(np.random.normal(0, 0.1, size=N))

        gaussian.GaussianEstimator(y=y, yerr=yerr, sigma_upper_bound=10)