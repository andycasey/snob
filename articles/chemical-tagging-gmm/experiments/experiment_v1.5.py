

import numpy as np
from astropy.table import Table
from sklearn import mixture
#from snob import nips_search3 as snob
from snob import mixture_ka as snob


np.random.seed(42)

try:
    catalog
except NameError:
    catalog = Table.read("../../../apogee-dr14-catalog.fits")

else:
    print("WARNING: USING PRE LOADED CATALOG")


# Number of Monte-Carlo realisations to do for each number of true clusters
M = 10
min_clusters = 1
#predictors = ("RA", "DEC", "VHELIO_AVG")

for element in ("NI", "O", "NA", "MG", "CA", "AL"):
    catalog["{}_FE".format(element)] = catalog["{}_H".format(element)] - catalog["FE_H"]

predictors = ("FE_H", "O_FE", "CA_FE", "MG_FE", )
covariance_type = "full"

D = len(predictors)

homoskedastic_uncertainty = 0.05
fast = False

min_membership_probability = 1.0
min_stars_per_cluster = 10

cluster_names = sorted(set(catalog["FIELD"][catalog["ASSOCIATION_PROB"] > 0]))


finite = np.all(np.array([
    np.isfinite(catalog[predictor]) * (catalog[predictor] > -10) \
    for predictor in predictors]), axis=0)

skip_cluster_names = []
for cluster_name in cluster_names:

    num = finite * (catalog["FIELD"] == cluster_name) \
        * (catalog["ASSOCIATION_PROB"] >= min_membership_probability)

    num = num.sum()
    print(cluster_name, num)

    if num < min_stars_per_cluster:
        print("Skipping {} because {} < {}".format(cluster_name, num, min_stars_per_cluster))
        skip_cluster_names.append(cluster_name)

cluster_names = sorted(set(cluster_names).difference(skip_cluster_names))

N = len(cluster_names)

print("Number of clusters ({}): {}".format(N, ", ".join(cluster_names)))


results = []


running_delta_aic = 0
running_delta_bic = 0
running_delta_mml = 0

fake_stars_per_cluster = 100

for n in range(1, 1 + N):

    for m in range(M):
        if m > 0 and n == N: break # because at n = N, we get all clusters.

        # Pick which clusters.
        selected_cluster_names = np.random.choice(
            cluster_names, size=n, replace=False)


        y = np.zeros((n * fake_stars_per_cluster, len(predictors)))
        true_mu = np.zeros((n, D))
        true_cov = np.zeros((n, D, D))
        true_cov_diag = np.zeros((n, D))

        for i, cluster_name in enumerate(selected_cluster_names):

            # Get all the stars from those clusters.
            match = finite \
                  * (catalog["FIELD"] == cluster_name) \
                  * (catalog["ASSOCIATION_PROB"] >= min_membership_probability)

            values = np.array([catalog[p][match] for p in predictors])

            mu = np.median(values, axis=1)
            cov = homoskedastic_uncertainty**2 * np.eye(D)

            si, ei = (i * fake_stars_per_cluster, (i + 1) * fake_stars_per_cluster)
            y[si:ei, :] = np.random.multivariate_normal(mu, cov, size=fake_stars_per_cluster)

            true_mu[i] = mu
            true_cov[i] = cov
            true_cov_diag[i] = homoskedastic_uncertainty**2

        true_weight = np.ones(n, dtype=float)/n

        # Construct the matrix of data.
        #y = np.array([catalog[p][match] for p in predictors]).T
        #y[:, 1] = y[:, 1] - y[:, 0]
        
        # Determine number of Gaussians from MML
        #model = snob.GaussianMixture(
        #    covariance_type=covariance_type, predict_mixtures=1,
        #    covariance_regularization=1e-6)
        #mu, cov, weight, meta = model.fit(y)
        
        #mml_num = weight.size

        # Just check,....
        #if mml_num != n:

        """
            dumb_check = False
            for zz in range(30):
                alt_model2 = snob.jump_to_mixture(y, n,
                    covariance_type=covariance_type,
                    covariance_regularization=1e-6,
                    threshold=1e-5, max_em_iterations=10000)
                if not abs(meta["message_length"] - alt_model2[-1]) < 1:
                    dumb_check = True

                    print("GOT A BETTER ONE FROM K = {} ({} < {}; {})".format(
                        n, alt_model2[-1], meta["message_length"], meta["message_length"] - alt_model2[-1]))

                    break
        """


        model = snob.GaussianMixture(covariance_regularization=1e-6,
            covariance_type=covariance_type)
        op_mu, op_cov, op_weight, meta = model.fit(y)
        mml_num = op_weight.size

        try:
            R, nll, true_ml = snob._expectation(y, true_mu, true_cov_diag if covariance_type == "diag" else true_cov, true_weight,
                covariance_type=covariance_type, covariance_regularization=1e-6)
        except:
            print("Failed to check truth")
            None

        else:


            if true_ml < meta["message_length"]:
                print("TRUE ML BETTER BY {}".format(true_ml - meta["message_length"]))

            
        """
        if fast:
            model = snob.GaussianMixture(
                covariance_type=covariance_type, predict_mixtures=10,
                covariance_regularization=1e-6)
            mu, cov, weight, meta = model.fit(y)
            mml_num = weight.size

        else:

            mls = []
            for ni in range(1, n+5):

                min_mls = []
                for zz in range(30):
                    alt_model = snob.jump_to_mixture(y, ni,
                        covariance_type=covariance_type,
                        covariance_regularization=1e-6,
                        threshold=1e-5, max_em_iterations=10000)
                    min_mls.append(alt_model[-1])

                mls.append([ni, np.min(min_mls)])

            mls = np.array(mls)
            _ = np.argmin(mls.T[1])

            if _ == (mls.shape[0] - 1):
                raise a
            #print("FROM {} to {}".format(mml_num, mls[_][0]))

            mml_num, __ = mls[_]
        """



        """
        # Consider alternative MML, where we initialize it at the true solution
        model2 = snob.GaussianMixture(
            covariance_type=covariance_type, covariance_regularization=1e-6)
        op_mu2, op_cov2, op_weight2, meta2 = model.fit(y,
            __initialize=(true_mu, true_cov, true_weights))

        if op_weight2.size != op_weight.size:
            #assert meta2["message_length"] < meta["mess age_length"]
            print("DID BETTER FROM TRUE")
            mml_num = op_weight2.size
        """

        # Consider alternative where we initialize at K = XX?


        # Determine number of components by AIC/BIC.
        aic = []
        bic = []
        aic_converged = -1
        bic_converged = -1
        for k in range(1, 1 + 2*N):
            try:
                model = mixture.GaussianMixture(n_components=k, covariance_type=covariance_type, )
                fitted_model = model.fit(y)

            except ValueError:
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

        #mml_num = np.nan
        results.append((n, m, y.shape[0], aic_converged, bic_converged, mml_num))

        print(results[-1])
        running_delta_aic += abs(aic_converged - n)
        running_delta_mml += abs(mml_num - n)
        running_delta_bic += abs(bic_converged - n)
        print("MML/BIC/AIC", running_delta_mml, running_delta_bic, running_delta_aic)

results = np.array(results)

import pickle

with open("v1.5-fake-apogee-results-{}-{}.pkl".format(covariance_type,
    "fast" if fast else "slow"), "wb") as fp:
    pickle.dump(results, fp, -1)

diff_mml = np.sum(np.abs(results.T[0] - results.T[-1]))
diff_bic = np.sum(np.abs(results.T[0] - results.T[-2]))
diff_aic = np.sum(np.abs(results.T[0] - results.T[-3]))


fig, axes = plt.subplots(3)
offset = 3

for i, (ax, name) in enumerate(zip(axes, ("AIC", "BIC", "MML"))):
    ax.scatter(results.T[0], results.T[i + offset] - results.T[0], alpha=0.1, marker='s', s=100)

    ax.set_ylabel(r"$\Delta${}".format(name))
    #if not ax.is_last_row():
    #    ax.set_xticks()

