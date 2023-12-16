import numpy as np
import matplotlib.pyplot as plt
def plotClusters(X, K, R, soft=True):
    
    if soft:
        random_colors = np.random.random((K, 3))
        colors = R.dot(random_colors)
    else:
        colors = np.argmax(R,axis=1)
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.show()

