
"""
Simplest SLF model.
"""

import numpy as np
import matplotlib.pyplot as plt

from snob.slf_mixture.single import SLFModel 

random_seed = 42
np.random.seed(42)

# Generate some data.
D = 2
N = 500

means = np.random.uniform(low=-1, high=1, size=D)
factor_scores = np.random.normal(0, size=N).reshape((N, 1))
factor_loads = np.random.uniform(low=-10, high=10, size=D).reshape((1, D))
specific_variances = np.random.normal(0, 0.5, size=D)**2
variates = np.random.normal(0, 1, size=(N, D))

X = means + np.dot(factor_scores, factor_loads) + specific_variances * variates

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
ax.scatter(specific_variances**0.5, mml_mod.specific_variances**0.5)
_common_limits(ax)
ax.set_title('specific_sigmas')




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

