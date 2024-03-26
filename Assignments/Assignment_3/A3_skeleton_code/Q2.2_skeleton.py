# Import useful libraries. Feel free to use sklearn.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Construct a 2D toy dataset for clustering.
X, _ = make_blobs(n_samples=1000,
                  centers=[[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]],
                  cluster_std=[0.2, 0.3, 0.3, 0.3, 0.3],
                  random_state=26)

# Conduct clustering on X using k-Means, and determine the best k with the elbow method.
def plot_curve(X, max_k=10):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(
            kmeans.inertia_
        )  # Sum of squared distances to closest cluster center

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), distortions, marker="o")
    plt.title("Curve")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Sum of Squared Distances")
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()