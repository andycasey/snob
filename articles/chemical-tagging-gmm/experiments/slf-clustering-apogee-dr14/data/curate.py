
""" Curate a sample of known clusters in APOGEE. """

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

data = Table.read("apogee-dr14-cannon.fits")

# Identify cluster candidates by field.
cluster_field_names = {
    'M12-N': dict(VHELIO_AVG=(-30, -50), FE_H=(-1.1, -1.7)),
    'M13': dict(VHELIO_AVG=(-220, -270), FE_H=(-1.3, -2.0)),
    'M15': dict(VHELIO_AVG=(-85, -120), FE_H=(-1.9, -2.4)),
    'M2': dict(VHELIO_AVG=(-20, 10), FE_H=(-1.8, -1.3)),
    'M3': dict(VHELIO_AVG=(-130, -160), FE_H=(-2, -1.3)),
    'M5': dict(VHELIO_AVG=(40, 65), FE_H=(-1.8, -1.0)),
    'M53': dict(VHELIO_AVG=(35, 50), FE_H=(-2.2, -1.95)),
    'M54SGRC1': dict(VHELIO_AVG=(180, 110), FE_H=(0, -1.75)),
    'M5PAL5': dict(VHELIO_AVG=(40, 65), FE_H=(-1.8, -1.1)),
    'M67': dict(FE_H=(-0.5, 0.5)),
    'M71': dict(FE_H=(-1, -0.6), VHELIO_AVG=(-18, -28)),
    'M92': dict(VHELIO_AVG=(-105, -135), FE_H=(-1.90, -2.25)),
    'N5466': dict(VHELIO_AVG=(95, 115), FE_H=(-1.7, -2.15)),
    'N6791': dict(VHELIO_AVG=(-42, -52), FE_H=(0.3, 0.55)),
}

# For each cluster, identify members based on RV.
is_member_of_field_name = np.zeros(len(data), dtype=bool)

for cluster_field_name, selection_criteria in cluster_field_names.items():
    print(cluster_field_name)

    is_candidate = (data["FIELD"] == cluster_field_name)


    fig, ax = plt.subplots()

    ax.set_title(cluster_field_name)

    """
    # Or just as a mixture of two gaussians,...?
    X = np.array([
        data["VHELIO_AVG"][is_candidate],
        data["FE_H"][is_candidate]
    ]).T
    gmm = GaussianMixture(n_components=2, covariance_type="full")
    gmm.fit(X)

    for j, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        width, height = 2 * np.sqrt(vals)

        ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
            facecolor="r", alpha=0.5)
        ax.add_artist(ellipse)

        ax.scatter([mean[0]], [mean[1]], facecolor="r")

    assignments = gmm.predict(X)

    # Which is closer to the median RV?
    median_rv = np.nanmedian(data["VHELIO_AVG"][is_candidate])
    indices = np.argsort(np.abs(gmm.means_[:, 0] - median_rv))

    is_member = (assignments == indices[0])
    is_not_member = ~is_member
    """

    is_member = np.ones(sum(is_candidate), dtype=bool)
    if selection_criteria is not None:
        for key, values in selection_criteria.items():
            min_value = min(values)
            max_value = max(values)

            is_member *= (max_value >= data[key][is_candidate]) \
                       * (data[key][is_candidate] >= min_value)

    is_not_member = ~is_member

    ax.scatter(
        data["VHELIO_AVG"][is_candidate][is_member],
        data["FE_H"][is_candidate][is_member])

    ax.scatter(
        data["VHELIO_AVG"][is_candidate][is_not_member],
        data["FE_H"][is_candidate][is_not_member],
        facecolor="#666666", zorder=-1)
    

    is_member_of_field_name[is_candidate] = is_member


