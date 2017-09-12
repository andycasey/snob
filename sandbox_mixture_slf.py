import numpy as np
from snob import mixture_slf as slf



n_samples, n_features, n_clusters, rank = 1000, 50, 6, 1
sigma = 0.5
true_homo_specific_variances = sigma**2 * np.ones((1, n_features))


rng = np.random.RandomState(321)

U, _, _ = np.linalg.svd(rng.randn(n_features, n_features))
true_factor_loads = U[:, :rank].T


true_factor_scores = rng.randn(n_samples, rank)
X = np.dot(true_factor_scores, true_factor_loads)

# Assign objects to different clusters.
indices = rng.randint(0, n_clusters, size=n_samples)
true_weights = np.zeros(n_clusters)
true_means = rng.randn(n_clusters, n_features)
for index in range(n_clusters):
    X[indices==index] += true_means[index]
    true_weights[index] = (indices==index).sum()

true_weights = true_weights/n_samples

# Adding homoscedastic noise
bar = rng.randn(n_samples, n_features)
X_homo = X + sigma * bar

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.
X_hetero = X + rng.randn(n_samples, n_features) * sigmas
true_hetero_specific_variances = sigmas**2

data = X_hetero

model = slf.SLFGMM(n_clusters)
model.fit(data)


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
scatter_common(true_homo_specific_variances, model.specific_variances, "specific variances")

# means
# This one is tricky because the indices are not necessarily the same.
# So just take whichever is closest.
idx = np.zeros(n_clusters, dtype=int)
for index, true_mean in enumerate(true_means):
    distance = np.sum(np.abs(model._means - true_mean), axis=1) \
             + np.abs(model.weights.flatten()[index] - true_weights)
    idx[index] = np.argmin(distance)

assert len(idx) == len(set(idx))

true = true_means.flatten()
inferred = model._means[idx].flatten()

scatter_common(true, inferred, "means")


# Plot some data...

fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1], facecolor="g")

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
