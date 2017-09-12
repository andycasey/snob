

import numpy as np

from astropy.table import Table
from snob import mixture_ka as snob
from sklearn import mixture


catalog = Table.read("catalog.fits")


realisations = {}

# Number of Monte-Carlo realisations to do for each number of true clusters
M = 10

# Number of potential clusters.
N = len(set(catalog["group"])) - 1 # subtract 1 to ignore field stars
min_clusters = 1

covariance_type = "diag"

predictors = ("RA", "DEC", "VRAD")

results = []

np.random.seed(1234567890)

for n in range(min_clusters, N):

    for m in range(M):
        if m > 0 and n == N: break # because at n = N, we get all clusters.

        # Pick which clusters.
        cluster_indices = np.random.choice(range(N), size=n, replace=False)

        # Get all the stars from those clusters.
        match = np.in1d(catalog["group"], cluster_indices)

        # Construct the matrix of data.
        y = np.array([catalog[p][match] for p in predictors]).T


        K = cluster_indices.size
        D = y.shape[1]
        true_mu = np.empty((K, D))
        true_cov = np.empty((K, D, D))
        true_weights = np.empty(K)

        for k, cluster_index in enumerate(cluster_indices):
            _match = (catalog["group"] == cluster_index) 
            _data = np.array([catalog[p][_match] for p in predictors])
            true_mu[k] = np.mean(_data, axis=1)
            true_weights[k] = _match.sum()
            true_cov[k] = np.cov(_data)

        true_weights /= true_weights.sum()

        # Determine number of Gaussians from MML
        model = snob.GaussianMixture(covariance_type=covariance_type, covariance_regularization=0)
        op_mu, op_cov, op_weight, meta = model.fit(y)
        mml_num = op_weight.size

        # TODO: Check if better from true solution?
        """
        model2 = snob.GaussianMixture(covariance_type=covariance_type, covariance_regularization=1e-6)
        op_mu2, op_cov2, op_weight2, meta2 = model.fit(y,
            __initialize=(true_mu, true_cov, true_weights))

        if op_weight2.size != op_weight.size:
            #assert meta2["message_length"] < meta["mess age_length"]
            print("DID BETTER FROM TRUE")
            mml_num = op_weight2.size
        """

        # Determine number of components by AIC/BIC.
        aic = []
        bic = []
        aic_converged = -1
        bic_converged = -1
        for k in range(1, 1 + 2*N):
            try:
                model = mixture.GaussianMixture(n_components=k, covariance_type=covariance_type, )
                fitted_model = model.fit(y)

            except:
                print("FAILED ON GMM TEST {}".format(k))
                aic_converged = 1 + np.argmin(aic)
                bic_converged = 1 + np.argmin(bic)

                break

            bic.append(fitted_model.bic(y))
            aic.append(fitted_model.aic(y))

            if k > 2:
                if aic_converged < 0 and np.all(np.diff(aic[-3:]) > 0):
                    aic_converged = 1 + np.argmin(aic)

                if bic_converged < 0 and np.all(np.diff(bic[-3:]) > 0):
                    bic_converged = 1 + np.argmin(bic)

                if aic_converged >= 0 and bic_converged >= 0:
                    break


        results.append((n, m, aic_converged, bic_converged, mml_num))

        print(results[-1])


results = np.array(results)

import pickle

with open("experiment_v0_results_{}.pkl".format(covariance_type), "wb") as fp:
    pickle.dump(results, fp, -1)

diff_mml = np.sum(np.abs(results.T[0] - results.T[-1]))
diff_bic = np.sum(np.abs(results.T[0] - results.T[-2]))
diff_aic = np.sum(np.abs(results.T[0] - results.T[-3]))

print("MML", diff_mml)
print("BIC", diff_bic)
print("AIC", diff_aic)

fig, axes = plt.subplots(3)

for i, (ax, name) in enumerate(zip(axes, ("AIC", "BIC", "MML"))):
    ax.scatter(results.T[0], results.T[i + 2] - results.T[0], alpha=0.1, marker='s', s=100)

    ax.set_ylabel(r"$\Delta${}".format(name))
    if not ax.is_last_row():
        ax.set_xticks()



raise a
