import glob
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def read_and_combine_distributions(file_pattern):
    combined_data = []
    image_names = []

    # Read all files matching the pattern
    for filename in glob.glob(file_pattern):
        with open(filename, 'rb') as f:
            score_dict = pickle.load(f)
            for image, distributions in score_dict.items():
                combined_data.append(distributions)
                image_names.append(image)
    
    return combined_data, image_names

def plot_pca_2d(data, image_names):
    # Flatten the distributions and create a feature matrix
    features = np.array([np.concatenate(distributions) for distributions in data])

    # Perform PCA to reduce dimensions to 2D
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(features)

    # Plot the 2D representation
    plt.figure(figsize=(10, 7))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], marker='o')
    for i, image_name in enumerate(image_names):
        plt.annotate(image_name, (transformed_data[i, 0], transformed_data[i, 1]))

    plt.title("PCA of Score Distributions")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Example usage
file_pattern = "score_distributions_*.pkl"
combined_data, image_names = read_and_combine_distributions(file_pattern)
plot_pca_2d(combined_data, image_names)
