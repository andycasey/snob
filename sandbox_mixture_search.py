
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



y = np.loadtxt("cluster_example.txt")


#n_samples = 500

# Generate random sample, two components
#np.random.seed(0)
#C = np.array([[0., -0.1], [1.7, .4]])
#X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
#          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
#y = X


#from sklearn import datasets
#iris = datasets.load_iris()
#y = iris.data

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

search_model = mixture.GaussianMixture()
search_model.search(y, 10)



model1 = mixture.GaussianMixture()
mu_1, cov_1, weight_1, meta_1 = model1.fit(y, 1)


model2 = mixture.GaussianMixture()
mu_2, cov_2, weight_2, meta_2 = model2.fit(y, 2)

N, D = y.shape
K = 1

Q_K = (0.5 * D * (D + 3) * K) + (K - 1)
Q_K2 = (0.5 * D * (D + 3) * (K + 1)) + (K + 1 - 1)

# Calculate the deltas in message length, according to  our expression.
actual_delta_I = meta_2["message_length"] - meta_1["message_length"]
expected_delta_I = np.log(2) \
    + np.log(N)/2.0 - np.log(K) - 0.5 * (np.sum(np.log(weight_2)) - np.sum(np.log(weight_1))) \
    - D * np.log(2)/2.0 + D * (D+3)/4.0 * (np.log(N) + np.sum(np.log(weight_2)) - np.sum(np.log(weight_1))) - (D + 2)/2.0 * (np.sum(np.log(np.linalg.det(cov_2))) - np.sum(np.log(np.linalg.det(cov_1)))) \
    + 0.25 * (2 * np.log(Q_K2/Q_K) - (D * (D+3) + 2) * np.log(2*np.pi)) \
    + meta_2["log_likelihood"] - meta_1["log_likelihood"]
expected_delta_I = expected_delta_I/np.log(2)


# OK,. let's see if we can estimate the learning rate \gamma
def _evaluate_gaussian(y, mu, cov):
   N, D = y.shape
   Cinv = np.linalg.inv(cov)
   scale = 1.0/np.sqrt((2*np.pi)**D * np.linalg.det(cov))#
   #Cinv**(-0.5)
   d = y - mu
   return scale * np.exp(-0.5 * np.sum(d.T * np.dot(Cinv, d.T), axis=0))

other = np.log(2) \
    + np.log(N)/2.0 - np.log(K) - 0.5 * (np.sum(np.log(weight_2)) - np.sum(np.log(weight_1))) \
    - D * np.log(2)/2.0 + D * (D+3)/4.0 * (np.log(N) + np.sum(np.log(weight_2)) - np.sum(np.log(weight_1))) \
    - (D + 2)/2.0 * (np.sum(np.log(np.linalg.det(cov_2))) - np.sum(np.log(np.linalg.det(cov_1)))) \
    + 0.25 * (2 * np.log(Q_K2/Q_K) - (D * (D+3) + 2) * np.log(2*np.pi))

gamma = K * _evaluate_gaussian(y, mu_1[0], cov_1[0]).sum() * (actual_delta_I - other)

# OK, now use gamma to estimate K = 3

K = 2
Q_K3 = (0.5 * D * (D + 3) * K) + (K - 1)

# Let us assume the determinants of covariance matrices will decrease:
cov_3_est = K / (K + 1) * np.linalg.det(cov_2)
cov_3_est = np.hstack([cov_3_est.min(), cov_3_est])

est_weight_3 = np.array([1/3., 1/3., 1/3.])


I_K3_to_K2 = np.log(2) \
    + np.log(N)/2.0 - np.log(K) - 0.5 * (np.sum(np.log(est_weight_3)) - np.sum(np.log(weight_2))) \
    - D * np.log(2)/2.0 + D * (D+3)/4.0 * (np.log(N) + np.sum(np.log(est_weight_3)) - np.sum(np.log(weight_2))) \
    - (D + 2)/2.0 * (np.sum(np.log(cov_3_est)) - np.sum(np.log(np.linalg.det(cov_2)))) \
    + 0.25 * (2 * np.log(Q_K3/Q_K2) - (D * (D+3) + 2) * np.log(2*np.pi)) \
    + gamma/(K+1) * np.sum(weight_2 * np.vstack([_evaluate_gaussian(y, mu_2[i], cov_2[i]) for i in range(2)]).T)


raise a

delta_I = np.log(2) + 0.5 * np.log(N) - np.log(K) \
        + 0.5 * (D*(D+3)/2.0 * np.log(N) - D * np.log(2)) \
        + 0.5 * (np.sum(np.log(np.linalg.det(cov_2))) - np.sum(np.log(np.linalg.det(cov_1)))) \
        + 0.5 * (np.log(Q_K2/Q_K) - np.log(2*np.pi)/2.0 * (D * (D + 3) + 2))

raise a