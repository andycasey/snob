
"""
Do chemical tagging on toy data to check the SLF convergence.
"""

import numpy as np
import pickle
from astropy.table import Table
import matplotlib.pyplot as pltmodel

import utils
from snob.slf_mixture import (MLMixtureModel, MMLMixtureModel)

random_seed = 42

K_start = 1
M = 10 # number of monte carlo realisations to do at each K


USE_TOY_DATA = True

prefix = "toy_" if USE_TOY_DATA else ""
catalog = Table.read("data/{}catalog.fits".format(prefix))
with open("data/{}generated-parameters.pkl".format(prefix), "rb") as fp:
    gp = pickle.load(fp)




X = np.vstack([catalog[name] for name in catalog.dtype.names[2:]]).T
K_clusters_total = len(set(catalog["cluster_id"]))

#keep = catalog["cluster_id"] < 10
#X = X[keep]


mml_mod = MMLMixtureModel(num_components=1, initialization_method="kmeans++")
mml_mod.fit(X)


def _common_limits(ax):
    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (np.min(limits), np.max(limits))

    ax.plot(limits, limits, c="#666666", lw=1, linestyle=":", zorder=-1)

    ax.set_xlim(limits)
    ax.set_ylim(limits)


fig, ax = plt.subplots()
ax.scatter(gp["factor_loads"], mml_mod.factor_loads)
_common_limits(ax)
ax.set_title('factor loads')


fig, ax = plt.subplots()
ax.scatter(gp["factor_scores"], mml_mod.approximate_factor_scores)
_common_limits(ax)
ax.set_title('factor scores')

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

