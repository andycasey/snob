import numpy as np
from snob import mixture_slf as slf


#y = np.loadtxt("cluster_abundances.txt")

n_samples, n_features, n_clusters, rank = 1000, 50, 2, 1
sigma = 0.5
true_specific_variances = sigma**2 * np.ones((1, n_features))


rng = np.random.RandomState(100)

U, _, _ = np.linalg.svd(rng.randn(n_features, n_features))
true_factor_loads = U[:, :rank].T


true_factor_scores = rng.randn(n_samples, rank)
X = np.dot(true_factor_scores, true_factor_loads)

# Assign objects to different clusters.
indices = rng.randint(0, n_clusters, size=n_samples)
#mu = 10 * np.arange(n_clusters)#rng.randn(n_clusters, n_features)
true_means = rng.randn(n_clusters, n_features)
for index in range(n_clusters):
    X[indices==index] += true_means[index]

# Adding homoscedastic noise
bar = rng.randn(n_samples, n_features)
X_homo = X + sigma * bar

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.
X_hetero = X + rng.randn(n_samples, n_features) * sigmas


#y = np.atleast_2d(np.loadtxt("coffee_example.txt")).T

#true_means = np.zeros((1, n_features))

model = slf.SLFGaussianMixture(2)
model.initialize_parameters(X_homo)

#model.fit(X_homo)
fig, axes = plt.subplots(3)

# means
# This one is tricky because the indices are not necessarily the same.
# So just take whichever is closest.
indices = np.zeros(n_clusters, dtype=int)
for index, true_mean in enumerate(true_means):
    distance = np.sum(np.abs(model._means - true_mean), axis=1)
    indices[index] = np.argmin(distance)

assert len(indices) == len(set(indices))

ax = axes[0]
true = true_means.flatten()
inferred = model._means[indices].flatten()
ax.scatter(true, inferred)


# factor scores
ax = axes[1]
true = true_factor_scores.flatten()
inferred = model._factor_scores.flatten()
ax.scatter(true, inferred)

# factor loads
ax = axes[2]
true = true_factor_loads.flatten()
inferred = model._factor_loads.flatten()
ax.scatter(true, inferred)




raise a

true = np.hstack([each.flatten() for each in (true_means, true_factor_scores, true_factor_loads, true_specific_variances)])

inferred = np.hstack([each.flatten() for each in (model.means, model.factor_scores, model.factor_loads, model.specific_variances)])

fig, ax = plt.subplots()
ax.scatter(true, inferred, alpha=0.5)

raise a
