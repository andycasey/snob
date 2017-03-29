

""" Test the mixture of Gaussians estimator. """

import numpy as np
import unittest

from .. import mixture


class TestIntegration(unittest.TestCase):


    def _generate_data(self):
        # Generate some data.
        N = 900
        fractions = np.ones(3)/3.0
        mu = np.array([
            [0, 0],
            [0, 2],
            [0, -2]
        ])
        cov = np.array([
            [
                [2, 0],
                [0, 0.2]
            ],
            [
                [2, 0],
                [0, 0.2],
            ],
            [
                [2, 0],
                [0, 0.2]
            ]
        ])

        y = np.reshape([np.random.multivariate_normal(
                mu[i], cov[i], size=int(N * fractions[i])) \
            for i in range(len(fractions))], (-1, 2))
        return y


    def test_three_component(self):
        """
        Perform the same 3-component test described in Figueiredo and Jain (2002).
        """

        y = self._generate_data()

        model = mixture.GaussianMixtureEstimator(y, 25)
        (op_mu, op_cov, op_fractions), ll = model.optimize()

        self.assertEqual(op_fractions.size, 3)


class TestSKLearnExample(unittest.TestCase):

    def test_sklearn(self):

        # From http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py

        n_samples = 500

        # Generate random sample, two components
        np.random.seed(0)
        C = np.array([[0., -0.1], [1.7, .4]])
        X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
                  .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

        model = mixture.GaussianMixtureEstimator(X, 10)
        (op_mu, op_cov, op_fractions), ll = model.optimize()


        #self.assertEqual(op_fractions.size, 2)



    def test_iris(self):

        # http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py

        from sklearn import datasets

        iris = datasets.load_iris()

        model = mixture.GaussianMixtureEstimator(iris.data, 10)
        (op_mu, op_cov, op_fractions), ll = model.optimize()

        #self.assertEqual(op_fractions.size, 3)

