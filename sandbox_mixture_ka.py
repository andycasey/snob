import numpy as np
from snob import mixture_ka as mixture

# Generate some reproducibly random data.
np.random.seed(1)

N = 900
fractions = np.ones(3)/3.0
mu = np.array([
    [0, 0, 0],
    [0, 2, 1],
    [0, -2, -1]
])
cov = np.array([
    [
        [2, 0, 0],
        [0, 0.2, 0],
        [0, 0, 0.05**2]
    ],
    [
        [2, 0, 0],
        [0, 0.2, 0],
        [0, 0, 0.05**2]
    ],
    [
        [2, 0, 0],
        [0, 0.2, 0],
        [0, 0, 0.05**2]
    ]
])

y = np.reshape([np.random.multivariate_normal(
        mu[i], cov[i], size=int(N * fractions[i])) \
    for i in range(len(fractions))], (-1, 3))


model = mixture.GaussianMixture(y)
mu, cov, weight = model.fit()



# Generate data from the example in Section 13.4 of P & A (2015)
np.random.seed(1)
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
