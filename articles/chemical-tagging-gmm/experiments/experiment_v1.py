

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


predictors \
    = ("FE1", "MG1", "TI1", "SI1", "CA1",)# "AL1", "CR1", "MN1", "NI1", "CO1")#, "BA2")

predictors = ("FE1", "MG1", )#"TI1", "SI1", "CA1", "AL1", "CR1", "BA2")


finite = np.all(np.array([
    np.isfinite(catalog[predictor]) for predictor in predictors]), axis=0)

results = []

np.random.seed(1234567890)

for n in range(min_clusters, N):

    for m in range(M):
        if m > 0 and n == N: break # because at n = N, we get all clusters.

        # Pick which clusters.
        cluster_indices = np.random.choice(range(N), size=n, replace=False)

        # Get all the stars from those clusters.
        match = finite * np.in1d(catalog["group"], cluster_indices)

        # Construct the matrix of data.
        y = np.array([catalog[p][match] for p in predictors]).T
        
        raise a

        # Determine number of Gaussians from MML
        model = snob.GaussianMixture(y)
        op_mu, op_cov, op_weight, meta = model.fit(covariance_regularization=0)
        mml_num = op_weight.size

        # Consider alternative MML, where we initialize it at the true solution

        model2 = snob.GaussianMixture(y)
        op_mu2, op_cov2, op_weight2, meta2 = model.fit(
            covariance_regularization=0,
            __initialize=)
    
        # Determine number of components by AIC/BIC.
        aic = []
        for k in range(1, 1 + 2*N):
            model = mixture.GaussianMixture(n_components=k)
            fitted_model = model.fit(y)

            aic.append(fitted_model.aic(y))

            if k > 2 \
            and np.all(np.diff(aic[-3:]) > 0):
                break

        best_by_aic = 1 + np.argmin(aic)


        # Determine number of components by BIC.
        bic = []
        for k in range(1, 1 + 2*N):
            model = mixture.GaussianMixture(n_components=k)
            fitted_model = model.fit(y)

            bic.append(fitted_model.bic(y))

            if k > 2 \
            and np.all(np.diff(bic[-3:]) > 0):
                break

        best_by_bic = 1 + np.argmin(bic)

        results.append((n, m, y.shape[0], best_by_aic, best_by_bic, mml_num))

        print(results[-1])

results = np.array(results)

import pickle

with open("experiment_v1_results.pkl", "wb") as fp:
    pickle.dump(results, fp, -1)

diff_mml = np.sum(np.abs(results.T[0] - results.T[-1]))
diff_bic = np.sum(np.abs(results.T[0] - results.T[-2]))
diff_aic = np.sum(np.abs(results.T[0] - results.T[-3]))

print("MML", diff_mml)
print("BIC", diff_bic)
print("AIC", diff_aic)

fig, axes = plt.subplots(3)

for i, ax in enumerate(axes):
    ax.scatter(results.T[0], results.T[i + 3], alpha=0.25, marker='s', s=100)

    ax.set_ylim(0, 10)
    ax.set_xlim(0, 20)

raise a
