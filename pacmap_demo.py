import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import pacmap
import numpy as np
from scipy.spatial import ConvexHull


def plot_pca_by_onehot_trait(X_transformed, data, trait_prefix):
    # Filter columns for the specific trait
    trait_columns = [col for col in data.columns if col.startswith(trait_prefix)]
    
    # Plotting
    plt.figure(figsize=(6, 6))
    for col in trait_columns:
        # Extract the category name without the prefix
        category_name = col[len(trait_prefix):]
        # Only plot the points where the trait value is 1
        indices = data[col] == 1
        plt.scatter(X_transformed[indices, 0], X_transformed[indices, 1], label=category_name, s=4.0)
    
    plt.title(f'PCA Embedding Colored by {trait_prefix}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()


# Function to extract the outer shell of PCA-transformed data
def extract_outer_shell(X_transformed):
    # Calculate the convex hull of the points
    hull = ConvexHull(X_transformed)
    
    # Get the indices of the points on the convex hull
    indices_on_hull = hull.vertices
    
    # Select the points on the convex hull
    outer_shell_points = X_transformed[indices_on_hull]
    
    return outer_shell_points, indices_on_hull


# Load your data (make sure to exclude any non-feature columns like 'userId')
data = pd.read_csv('OneHotEncoded_Traits_Corrected.csv')
X = data.drop('userId', axis=1)

# Initialize PCA instance for 2 components
pca = PCA(n_components=2)

embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 

# fit the data (The index of transformed data corresponds to the index of the original data)
X_transformed = embedding.fit_transform(X, init="pca")

# Fit and transform the data using PCA
# X_transformed = pca.fit_transform(X)

# plot_pca_by_onehot_trait(X_transformed, data, 'gender_')
# plot_pca_by_onehot_trait(X_transformed, data, 'age_')
# plot_pca_by_onehot_trait(X_transformed, data, 'EducationalLevel_')
# plot_pca_by_onehot_trait(X_transformed, data, 'artExperience_')
# plot_pca_by_onehot_trait(X_transformed, data, 'photographyExperience_')


# Extract the outer shell of the PCA-transformed data and the shell indices
outer_shell, shell_indices = extract_outer_shell(X_transformed)

# Calculate distances between each data point and all shell vertices
distances = np.linalg.norm(X_transformed[:, np.newaxis] - outer_shell, axis=2)

# Find the indices of the closest shell vertex for each data point
closest_vertex_distance = np.min(distances, axis=1)
closest_vertex_indices = np.argsort(closest_vertex_distance)[:50]

# Select the corresponding user IDs and data points for the unique closest vertices
unique_closest_user_ids = data['userId'][closest_vertex_indices]
unique_closest_data_points = X_transformed[closest_vertex_indices]


# Save the unique closest user IDs to a CSV file
unique_closest_user_ids_df = pd.DataFrame(unique_closest_user_ids, columns=['userId'])
unique_closest_user_ids_df.to_csv('shell_user_ids.csv', index=False)



# Plot the extracted outer shell
plt.figure(figsize=(6, 6))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=0.6)
plt.scatter(outer_shell[:, 0], outer_shell[:, 1], s=10, color='red', label='Outer Shell')
plt.title('PCA Embedding with Outer Shell')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Visualize the 100 unique data points
plt.figure(figsize=(8, 8))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=0.6, label='All Data')
plt.scatter(unique_closest_data_points[:, 0], unique_closest_data_points[:, 1], s=10, color='red', label='Unique Closest Data')
plt.title('Visualization of 100 Unique Closest Data Points')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

plt.show()