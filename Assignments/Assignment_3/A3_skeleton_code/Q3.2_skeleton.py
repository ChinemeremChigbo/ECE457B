# Import useful libraries
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# Convert labels to integers
y = y.astype(int)

# Filter out digits '0', '1', and '3'
digits_to_keep = [0, 1, 3]
X_filtered = X[np.isin(y, digits_to_keep)]
y_filtered = y[np.isin(y, digits_to_keep)]

# Conduct PCA to reduce the dimensionality of X
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_filtered)

# Visualize the data distribution of digits '0', '1' and '3' in a 2D scatter plot
plt.figure(figsize=(10, 6))
for digit in digits_to_keep:
    plt.scatter(
        X_pca[y_filtered == digit, 0], X_pca[y_filtered == digit, 1], label=str(digit)
    )
plt.title("MNIST Data Distribution of Digits 0, 1, and 3 (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()

# Generate an image of digit '3' using 2D representations of digits '0' and '1'
# Find indices of digit '3' in the filtered data
index_digit_3 = np.where(y_filtered == 3)[0][0]

# Find the corresponding 2D representation of digit '3'
x_3 = X_pca[index_digit_3]

# Find the closest digit '0' and '1' to the representation of digit '3'
index_digit_0 = np.argmin(np.linalg.norm(X_pca[y_filtered == 0] - x_3, axis=1))
index_digit_1 = np.argmin(np.linalg.norm(X_pca[y_filtered == 1] - x_3, axis=1))

# Plot the original digits
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.imshow(X_filtered[y_filtered == 0][index_digit_0].reshape(28, 28), cmap="gray")
plt.title("Digit 0")

plt.subplot(1, 3, 2)
plt.imshow(X_filtered[y_filtered == 1][index_digit_1].reshape(28, 28), cmap="gray")
plt.title("Digit 1")

plt.subplot(1, 3, 3)
plt.imshow(X_filtered[y_filtered == 3][index_digit_3].reshape(28, 28), cmap="gray")
plt.title("Digit 3 (Original)")
plt.show()
