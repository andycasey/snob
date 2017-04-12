import numpy as np
from snob import mixture_slf as slf


#y = np.loadtxt("cluster_abundances.txt")

n_samples, n_features, rank = 1000, 50, 1
sigma = 0.5
true_specific_variances = sigma**2 * np.ones((1, n_features))


rng = np.random.RandomState(100)

U, _, _ = np.linalg.svd(rng.randn(n_features, n_features))
true_factor_loads = U[:, :rank].T


true_factor_scores = rng.randn(n_samples, rank)
X = np.dot(true_factor_scores, true_factor_loads)

# Adding homoscedastic noise
bar = rng.randn(n_samples, n_features)
X_homo = X + sigma * bar

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.
X_hetero = X + rng.randn(n_samples, n_features) * sigmas


#y = np.atleast_2d(np.loadtxt("coffee_example.txt")).T

true_means = np.zeros((1, n_features))

model = slf.SLFGaussianMixture(1)
model.fit(X_homo)


raise a

true = np.hstack([each.flatten() for each in (true_means, true_factor_scores, true_factor_loads, true_specific_variances)])

inferred = np.hstack([each.flatten() for each in (model.means, model.factor_scores, model.factor_loads, model.specific_variances)])

fig, ax = plt.subplots()
ax.scatter(true, inferred, alpha=0.5)

raise a
