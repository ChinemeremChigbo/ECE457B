# Import useful libraries. Feel free to use sklearn.
from sklearn.datasets import fetch_openml


# Load MNIST dataset.
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# Conduct PCA to reduce the dimensionality of X.


# Visualize the data distribution of digits '0', '1' and '3' in a 2D scatter plot.


# Generate an image of digit '3' using 2D representations of digits '0' and '1'.
