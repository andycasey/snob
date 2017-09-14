
""" Generate a fake catalog of open cluster stars. """


# Assumptions:

# (0) Every cluster has a unique chemical fingerprint.

# (1) There is a underlying, single latent factor common to all abundances.

# (2) The abundances in every cluster star will have some intrinsic variance.

# (3) The abundances will have some uncertainty in their measurements, and
#     uncertainty is homoscedastic.

# (4) The number of stars in each cluster can be different.

# (5) The cluster abundances are taken from some mean metallicity from the
#     work of Chen et al. (2003): http://adsabs.harvard.edu/abs/2003AJ....125.1397C


import pickle
import numpy as np
from astropy.table import Table

random_seed = 1
np.random.seed(random_seed)

oc_mean_properties = Table.read("Chen-et-al-2003-AJ-125-1397-table1.fits")


# Adjustable parameters #
# --------------------- #
elements_measured = ("Fe", "Ti", "Ca", "Si")
elements_measured = "ABCDEFGHIJKLMNO"

def draw_number_of_stars_in_cluster(mean_cluster_abundance=None):
    return np.random.randint(1, 250)

means = np.mean(oc_mean_properties["__Fe_H_"]) \
      + np.random.normal(0, 0.1, size=len(elements_measured))

specific_sigmas = np.clip(
    np.abs(np.random.normal(0, 0.01, size=len(elements_measured))),
    0.005, 0.05)

cluster_dispersion = 0.01

magnitude_of_factor_load = 0.1
# --------------------- #


K = len(elements_measured)

# Generate the factor_loads (a_k)
factor_loads = np.random.uniform(
    -abs(magnitude_of_factor_load), 
    +abs(magnitude_of_factor_load),
    size=(1, K))

data = []
cluster_variates = []
cluster_factor_scores = []


for i, open_cluster in enumerate(oc_mean_properties):

    mean_cluster_abundance = open_cluster["__Fe_H_"]

    # How many stars will we draw from this cluster.
    N_cluster = draw_number_of_stars_in_cluster(mean_cluster_abundance)

    # Determine the cluster fingerprint.
    # factor_scores (v_n)
    factor_scores = np.random.normal(0, 1) * np.ones((N_cluster, 1))
    #+ np.random.normal(1, cluster_dispersion, 
    #  size=(N_cluster, 1))

    # variates (r_{nk})
    variates = np.random.normal(0, 1, size=(N_cluster, K))

    stellar_abundances = means \
                       + np.dot(factor_scores, factor_loads) \
                       + specific_sigmas * variates

    # TODO: add measurement noise.

    """
    _ = ax.plot(np.mean(means + np.dot(factor_scores, factor_loads), axis=0),
     lw=2)
    ax.plot(stellar_abundances.T, c=_[0].get_color(), zorder=-1, alpha=0.1)
    """

    # Generate rows for the faux catalog.
    rows = np.vstack([
        i * np.ones(N_cluster, dtype=int).T,
        stellar_abundances.T
    ]).T

    data.extend(rows)

    cluster_variates.append(variates)
    cluster_factor_scores.append(factor_scores)



# Generate a faux-catalog.
names = ["star_id", "cluster_id"]
names.extend(["{}_h".format(el).upper() for el in elements_measured])

# Collect the data together, and add a star_id column.
data = np.vstack(data)
N = len(data)
data = np.hstack([np.atleast_2d(np.arange(N)).T, data])

stars = Table(data, names=names)

for column_name in ("star_id", "cluster_id"):
    stars[column_name] = np.array(stars[column_name], dtype=int)

stars.write("catalog.fits", overwrite=True)


# Load the data that we may want to compare with later.
generated_parameters = dict(
    random_seed=random_seed,
    means=means,
    variates=np.vstack(cluster_variates),
    factor_loads=factor_loads,
    factor_scores=np.vstack(cluster_factor_scores),
    specific_sigmas=specific_sigmas
)

with open("generated-parameters.pkl", "wb") as fp:
    pickle.dump(generated_parameters, fp, -1)