

""" Test the mixture of Gaussians estimator. """

import numpy as np
import unittest

from .. import mixture_ka as mixture


class TestIntegration(unittest.TestCase):




    def test_four_component(self):
        """
        Perform the same four-component test described in Section 13.4 of
        P & A (2015).
        """

        np.random.seed(1)

        # Generate data from the example in Section 13.4 of P & A (2015)
        N = 1000
        weight = np.array([0.3, 0.3, 0.3, 0.1])
        mu = np.array([
            [-4, -4],
            [-4, -4],
            [2, 2],
            [-1, -6]
        ])

        cov = np.array([
            [
                [1, 0.5],
                [0.5, 1]
            ],
            [
                [6, -2],
                [-2, 6]
            ],
            [
                [2, -1],
                [-1, 2]
            ],
            [
                [0.125, 0],
                [0, 0.125]
            ]
        ])

        y = np.vstack([np.random.multivariate_normal(
                mu[i], cov[i], size=int(N * weight[i])) \
            for i in range(len(weight))])

        model = mixture.GaussianMixture(y)
        op_mu, op_cov, op_weight = model.fit()

        self.assertEqual(op_weight.size, 4)

        # Re-sort op_mu to match the order.

        indices = np.zeros(op_weight.size, dtype=int)
        for index in range(op_weight.size):
            euclidian_distance = np.sum(
                np.abs(cov - op_cov[index]).reshape(op_weight.size, -1), axis=1)
            indices[index] = np.argmin(euclidian_distance)

        # Sort optimized values.
        op_mu = op_mu[indices]
        op_cov = op_cov[indices]
        op_weight = op_weight[indices]

        self.assertTrue(np.allclose(mu, op_mu, 0.15))
        self.assertTrue(np.allclose(cov, op_cov, 0.40))
        self.assertTrue(np.allclose(weight, op_weight, 0.02))




'''
class TestSKLearnExample(unittest.TestCase):

    def test_sklearn(self):

        # From http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py

        n_samples = 500

        # Generate random sample, two components
        np.random.seed(0)
        C = np.array([[0., -0.1], [1.7, .4]])
        X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
                  .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

        model = mixture.GaussianMixture(X, 10)
        (op_mu, op_cov, op_fractions), ll = model.optimize()


        self.assertEqual(op_fractions.size, 2)



    def test_iris(self):

        # http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py

        from sklearn import datasets

        iris = datasets.load_iris()

        model = mixture.GaussianMixture(iris.data, 10)
        (op_mu, op_cov, op_fractions), ll = model.optimize()

        self.assertEqual(op_fractions.size, 3)

'''