
"""
Two-cluster model with a single latent factor.
"""

import numpy as np
import matplotlib.pyplot as plt

from snob.slf_mixture.single import SLFModel, _message_length

random_seed = 42
np.random.seed(random_seed)

# Generate some data.
D = 2
K = 2
N = 1000

means = np.random.uniform(low=-1, high=1, size=D)
specific_sigmas = np.abs(np.random.normal(0, 0.15, size=D))
variates = np.random.normal(0, 1, size=(N, D))

factor_loads = np.random.uniform(low=-1, high=1, size=D).reshape((1, D))

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


mml_mod = SLFModel()
mml_mod.fit(X)


def _common_limits(ax):
    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))

    ax.plot(limits, limits, c="#666666", lw=1, linestyle=":", zorder=-1)

    ax.set_xlim(limits)
    ax.set_ylim(limits)


fig, ax = plt.subplots()
ax.scatter(factor_loads, mml_mod.factor_loads)
_common_limits(ax)
ax.set_title('factor loads')


fig, ax = plt.subplots()
ax.scatter(factor_scores, mml_mod.factor_scores)
_common_limits(ax)
ax.set_title('factor scores')


fig, ax = plt.subplots()
ax.scatter(means, mml_mod.means)
_common_limits(ax)
ax.set_title('means')


fig, ax = plt.subplots()
ax.scatter(specific_sigmas, mml_mod.specific_variances**0.5)
_common_limits(ax)
ax.set_title('specific_sigmas')

I_inferred, inferred_comps = mml_mod.message_length(X, full_output=True)
I_truth, truth_comps = _message_length(X, means, factor_scores, factor_loads, specific_sigmas**2, full_output=True)

assert I_inferred <= I_truth


raise a




#ml_mod = MLMixtureModel(num_components=2)
#ml_mod.fit(X)


mml_translate_parameters = {"factor_scores": "approximate_factor_scores"}

for parameter_name in ("factor_scores", "factor_loads"):

    fig, ax = plt.subplots()

    x = gp[parameter_name].flatten()

    #ax.scatter(x, getattr(ml_mod, parameter_name).flatten(),
    #           facecolor="r", label="ml (aecm)")

    mml_parameter_name = mml_translate_parameters.get(
        parameter_name, parameter_name)

    ax.scatter(x, getattr(mml_mod, mml_parameter_name).flatten(),
               facecolor="b", label="mml (aecm)")

    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))

    ax.plot(limits, limits, c="#666666", lw=1, linestyle=":", zorder=-1)

    ax.set_xlim(limits)
    ax.set_ylim(limits)

    ax.set_title(parameter_name)

