# Import useful libraries. Feel free to use sklearn.
from sklearn.datasets import make_blobs


# Construct a 2D toy dataset for clustering.
X, _ = make_blobs(n_samples=1000,
                  centers=[[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]],
                  cluster_std=[0.2, 0.3, 0.3, 0.3, 0.3],
                  random_state=26)

# Conduct clustering on X using k-Means, and determine the best k with the elbow method.
