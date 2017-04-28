
import numpy as np
from snob import mixture_search as mixture



# Generate data from the example in Section 13.4 of P & A (2015)
np.random.seed(888)
N = 1000
weight = np.array([0.3, 0.3, 0.3, 0.1])
mu = np.array([
    [-4, -4],
    [-4, -4],
    [2, 2],
    [-1, -6]
])

cov = np.array([
    [
        [1, 0.5],
        [0.5, 1]
    ],
    [
        [6, -2],
        [-2, 6]
    ],
    [
        [2, -1],
        [-1, 2]
    ],
    [
        [0.125, 0],
        [0, 0.125]
    ]
])


y = np.vstack([np.random.multivariate_normal(
        mu[i], cov[i], size=int(N * weight[i])) \
    for i in range(len(weight))])

"""
Approximating \sum\log{w_k}...

bar = []
for i in range(1, 101):

    foo = np.random.uniform(size=(1000, i))
    foo = foo.T/np.sum(foo, axis=1)
    print(i)

    mean = np.mean(np.log(foo).sum(axis=0))
    std = np.std(np.log(foo).sum(axis=0))

    bar.append([i, mean, std])

    

bar = np.array(bar)

fig, ax = plt.subplots()
ax.scatter(bar.T[0], bar.T[2])
#ax.scatter(bar.T[0], bar.T[1])
#ax.errorbar(bar.T[0], bar.T[1], bar.T[2], fmt=None)

raise a
"""

"""
# Approximating \log(\sum{r}) (the log of the effective memberships...)
bar = []
for i in range(1, 101):

    foo = np.random.uniform(size=(1000, i))
    foo = foo.T/np.sum(foo, axis=1)
    foo *= 900 # The sample size, say.

    mean = np.mean(np.log(foo).sum(axis=0))
    std = np.std(np.log(foo).sum(axis=0))

    bar.append([i, mean, std])


bar = np.array(bar)

fig, axes = plt.subplots(2)
axes[0].scatter(bar.T[0], bar.T[1])
axes[1].scatter(bar.T[0], bar.T[2])

raise a
"""

model = mixture.GaussianMixture()
mu, cov, weight, meta = model.fit(y)


raise a