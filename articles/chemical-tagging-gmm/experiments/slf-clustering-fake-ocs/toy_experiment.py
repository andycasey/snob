
"""
Do chemical tagging on toy data to check the SLF convergence.
"""

import numpy as np
import pickle
from astropy.table import Table
import matplotlib.pyplot as pltmodel

import utils
from snob.slf_mixture import MLMixtureModel

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

model = MLMixtureModel(num_components=2)

model.fit(X)