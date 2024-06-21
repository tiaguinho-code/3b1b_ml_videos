import gensim.downloader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Query the user for a single line of input
user_input = input("Enter four words separated by spaces: ")

# Split the user input into a list of strings
words = user_input.split()

# Check if exactly four strings were provided
if len(words) == 4:
    print("You have entered:", words)
else:
    print("Error: Please enter exactly four strings.")

# Load the GloVe model
model = gensim.downloader.load("glove-wiki-gigaword-50")

# Get a sample of words from the model's vocabulary to fit the PCA
sample_words = list(model.index_to_key[:10000000])  # taking a sample of 1000 words for PCA fitting
vectors = np.array([model[word] for word in sample_words])

# Initialize PCA to reduce to 3 dimensions
pca = PCA(n_components=3)
pca.fit(vectors)

if len(words) != 4:
    raise ValueError("The 'words' list must contain exactly 4 words.")

# Prepare figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Variable to hold maximum absolute value for setting axes limits later
max_val = 0

# Transform and plot each word
reduced_vectors = []
for word in words:
    vector = model[word]
    reduced_vector = pca.transform(vector.reshape(1, -1))[0]
    reduced_vectors.append(reduced_vector)
    
    # Plot the vector from origin to the reduced point using quiver
    ax.quiver(0, 0, 0, reduced_vector[0], reduced_vector[1], reduced_vector[2], 
              label=word, arrow_length_ratio=0.1)
    ax.text(reduced_vector[0], reduced_vector[1], reduced_vector[2], f'{word}', color='red')
    
    # Update max_val for setting axis limits
    max_val = max(max_val, np.max(np.abs(reduced_vector)))

# Calculate and plot difference vectors
diff1 = reduced_vectors[1] - reduced_vectors[0]
diff2 = reduced_vectors[3] - reduced_vectors[2]

# Plot the difference vector from the position of the first word
ax.quiver(reduced_vectors[0][0], reduced_vectors[0][1], reduced_vectors[0][2], 
          diff1[0], diff1[1], diff1[2], color='blue', arrow_length_ratio=0.2)

# Plot the difference vector from the position of the third word
ax.quiver(reduced_vectors[2][0], reduced_vectors[2][1], reduced_vectors[2][2], 
          diff1[0], diff1[1], diff1[2], color='green', arrow_length_ratio=0.2)


# Set the limits of the axes
max_val *= 1.1  # Adding a little margin
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val])

# Set labels and title
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')
ax.set_title('3D plot of word vectors and difference vectors after PCA')

plt.show()
