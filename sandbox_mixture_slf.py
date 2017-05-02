import numpy as np
from snob import mixture_slf as slf


#y = np.loadtxt("cluster_abundances.txt")

n_samples, n_features, n_clusters, rank = 1000, 50, 3, 1
sigma = 1
true_homo_specific_variances = sigma**2 * np.ones((1, n_features))


rng = np.random.RandomState(321)

U, _, _ = np.linalg.svd(rng.randn(n_features, n_features))
true_factor_loads = U[:, :rank].T


true_factor_scores = rng.randn(n_samples, rank)
X = np.dot(true_factor_scores, true_factor_loads)

# Assign objects to different clusters.
indices = rng.randint(0, n_clusters, size=n_samples)
true_means = rng.randn(n_clusters, n_features)
for index in range(n_clusters):
    X[indices==index] += true_means[index]

# Adding homoscedastic noise
bar = rng.randn(n_samples, n_features)
X_homo = X + sigma * bar

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.
X_hetero = X + rng.randn(n_samples, n_features) * sigmas
true_hetero_specific_variances = sigmas**2

#y = np.atleast_2d(np.loadtxt("coffee_example.txt")).T

#true_means = np.zeros((1, n_features))

model = slf.SLFGaussianMixture(n_clusters)
#model.true_factor_scores = true_factor_scores
#model.true_factor_loads = true_factor_loads
#model.true_means = true_means
#model.true_specific_variances = true_specific_variances
model.fit(X_hetero)

#model.initialize_parameters(X_homo)

def scatter_common(x, y, title=None):
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    ax.set_title(title or "")
    limits = np.array([ax.get_xlim(), ax.get_ylim()])
    limits = (limits.min(), limits.max())
    ax.plot(limits, limits, c="#666666", linestyle=":", linewidth=0.5, zorder=-1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    
    return fig

scatter_common(true_factor_loads, model.factor_loads, "factor loads")
scatter_common(true_factor_scores, model.factor_scores, "factor scores")
scatter_common(true_hetero_specific_variances, model.specific_variances, "specific variances")

# means
# This one is tricky because the indices are not necessarily the same.
# So just take whichever is closest.
idx = np.zeros(n_clusters, dtype=int)
for index, true_mean in enumerate(true_means):
    distance = np.sum(np.abs(model._means - true_mean), axis=1)
    idx[index] = np.argmin(distance)

assert len(idx) == len(set(idx))

true = true_means.flatten()
inferred = model._means[idx].flatten()

scatter_common(true, inferred, "means")


# Plot some data...

fig, ax = plt.subplots()
ax.scatter(X_homo[:, 0], X_homo[:, 1], facecolor="g")

raise a

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
