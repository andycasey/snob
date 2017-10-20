
"""
Two-cluster model with a single latent factor.
"""

import numpy as np
import matplotlib.pyplot as plt

from snob.slf_mixture.single import SLFModel
from snob.slf_mixture import MMLMixtureModel


random_seed = 123
np.random.seed(random_seed)

# Generate some data.
D = 2
K = 2
N = 1000
load_magnitude = 5

means = np.random.uniform(low=-1, high=1, size=D)
specific_sigmas = np.abs(np.random.normal(0, 0.15, size=D))
variates = np.random.normal(0, 1, size=(N, D))

factor_loads = np.random.uniform(
    low=-load_magnitude, high=+load_magnitude, size=D).reshape((1, D))

weights = np.random.uniform(0, 1, size=K)
weights = weights/weights.sum()

effective_memberships = np.random.multinomial(N, weights)

factor_score_mu = np.random.normal(0, 1, size=K)
factor_score_sigma = np.abs(np.random.normal(0, 0.15, size=K))

factor_scores = np.zeros(N)
for i, (Nk, mu, sigma) in enumerate(zip(effective_memberships, factor_score_mu, factor_score_sigma)):

    si = effective_memberships[:i].sum()
    ei = si + Nk
    factor_scores[si:ei] = np.random.normal(mu, sigma, size=Nk)

factor_scores = factor_scores.reshape((N, 1))

X = means + np.dot(factor_scores, factor_loads) + specific_sigmas * variates

fig, ax = plt.subplots()
ax.scatter(X.T[0], X.T[1])

#single = SLFModel()
#single.fit(X)

# N = number of data points
# D = dimensions of data points
# L = number of latent factors
# K = number of clusters

# Correct shapes:
# factor_loads: (L, D)
# factor_scores: (N, L)
# specific_variances: (1, D)
# means: (1, D)


mixture = MMLMixtureModel(num_components=2)
mixture.fit(X)