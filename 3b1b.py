import gensim.downloader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the GloVe model
model = gensim.downloader.load("glove-wiki-gigaword-50")

# Get a sample of words from the model's vocabulary to fit the PCA
sample_words = list(model.index_to_key[:10000000])  # taking a sample of 1000 words for PCA fitting
vectors = np.array([model[word] for word in sample_words])

# Initialize PCA to reduce to 3 dimensions
pca = PCA(n_components=3)
pca.fit(vectors)

from mpl_toolkits.mplot3d import Axes3D

# Get the word vector for the word "tower" and transform it using PCA
vector = model["tower"]

reduced_vector = pca.transform(vector.reshape(1, -1))[0]

# Plot the reduced vector in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the vector from origin to the reduced point using quiver
ax.quiver(0, 0, 0, reduced_vector[0], reduced_vector[1], reduced_vector[2], 
          color='r', arrow_length_ratio=0.1)

ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.set_title('3D plot of the "tower" vector after PCA')
max_val = np.max(np.abs(reduced_vector)) * 1.1  # Adding a little margin
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val])
plt.show()
