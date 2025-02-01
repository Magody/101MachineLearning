import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

class GaussianMixtureModels:
    
    def __init__(self,X,K) -> None:
        self.initializeParameters(X,K)
        
    def initializeParameters(self,X,K):
        self.X = X
        self.K = K
        self.N, self.D = self.X.shape
        self.M = np.zeros((self.K, self.D)) # means
        self.C = np.zeros((self.K, self.D, self.D)) # covariance
        self.pi = np.ones(self.K) / self.K # uniform, pi
        
        
        # initialize M to random, initialize C to spherical with variance 1
        for k in range(self.K):
            self.M[k] = X[np.random.choice(self.N)]
            self.C[k] = np.eye(self.D)
        self.weighted_pdfs = np.zeros((self.N, self.K)) # we'll use these to store the PDF value of sample n and Gaussian k
        
    
    def cluster(self, max_iter=20, smoothing=1e-2):
        
        R = np.zeros((self.N, self.K))
        lls = []
        for i in range(max_iter):
            
            # step 1: "vectorization", calculate responsibilities
            for k in range(self.K):
                self.weighted_pdfs[:,k] = self.pi[k]*multivariate_normal.pdf(self.X, self.M[k], self.C[k])
            R = self.weighted_pdfs / self.weighted_pdfs.sum(axis=1, keepdims=True)

            # step 2: recalculate params
            for k in range(self.K):
                Nk = R[:,k].sum()
                self.pi[k] = Nk / self.N
                self. M[k] = R[:,k].dot(self.X) / Nk

                delta = self.X - self.M[k] # N x D
                Rdelta = np.expand_dims(R[:,k], -1) * delta # multiplies R[:,k] by each col. of delta - N x D
                self.C[k] = Rdelta.T.dot(delta) / Nk + np.eye(self.D)*smoothing # D x D
                

            ll = np.log(self.weighted_pdfs.sum(axis=1)).sum()
            lls.append(ll)
            if i > 0:
                if np.abs(lls[i] - lls[i-1]) < 0.1:
                    break
                
        return R, lls

"""
Example:
# assume 3 means
D = 2 # so we can visualize it more easily
s = 4 # separation so we can control how far apart the means are
mu1 = np.array([0, 0])
mu2 = np.array([s, s])
mu3 = np.array([0, s])

N = 2000 # number of samples
X = np.zeros((N, D))
X[:1200, :] = np.random.randn(1200, D)*2 + mu1
X[1200:1800, :] = np.random.randn(600, D) + mu2
X[1800:, :] = np.random.randn(200, D)*0.5 + mu3

K = 3
gmm = GaussianMixtureModels(X, K)
responsabilities, lls = gmm.cluster()
random_colors = np.random.random((K, 3))
colors = responsabilities.dot(random_colors)
print("responsabilities: ", responsabilities.shape)
print("random_colors: ", random_colors, colors.shape)
plt.scatter(X[:,0], X[:,1], c=colors)
plt.show()

print("pi:", gmm.pi)
print("means:", gmm.M)
print("covariances:", gmm.C)
"""