import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt



# Load the CSV file
evaluation_results_path = './evaluation_results.csv'
evaluation_results_df = pd.read_csv(evaluation_results_path)

# Remove duplicates based on UserId to ensure unique user representation
unique_users_df = evaluation_results_df.drop_duplicates(subset='UserId')

# Define the trait columns for PCA
trait_columns = [f'Trait_{i}' for i in range(1, 71)]

# Calculate the mean traits for the unique users
unique_users_trait_mean = unique_users_df[trait_columns].mean().values

# Calculate the Euclidean distance of each user's traits to the mean traits
unique_users_df['Distance_To_Mean_Trait'] = unique_users_df[trait_columns].apply(lambda x: np.linalg.norm(x.values - unique_users_trait_mean), axis=1)

# Perform 2D PCA on the traits for unique users
pca = PCA(n_components=2)
traits_pca = pca.fit_transform(unique_users_df[trait_columns])

# Compute the convex hull of the 2D PCA points
hull = ConvexHull(traits_pca)

# Calculate the distance of each user's PCA point to the convex hull vertices
def distance_to_hull(point, hull_vertices):
    return np.min([np.linalg.norm(point - vertex) for vertex in hull_vertices])

hull_vertices = traits_pca[hull.vertices]
unique_users_df['Distance_To_Hull'] = [distance_to_hull(point, hull_vertices) for point in traits_pca]

# Plot 1: Mean EMD Loss with respect to Distance to the Mean Trait
plt.figure(figsize=(10, 6))
plt.scatter(unique_users_df['Distance_To_Mean_Trait'], unique_users_df['EMD_Loss_Data'], alpha=0.5, label='Distance to Mean Trait')
plt.xlabel('Distance to Mean Trait')
plt.ylabel('Mean EMD Loss')
plt.title('Mean EMD Loss vs. Distance to Mean Trait')
plt.legend()

# Plot 2: Mean EMD Loss with respect to Distance to the Closest Convex Hull Vertex
plt.figure(figsize=(10, 6))
plt.scatter(unique_users_df['Distance_To_Hull'], unique_users_df['EMD_Loss_Data'], alpha=0.5, label='Distance to Convex Hull Vertex')
plt.xlabel('Distance to Closest Convex Hull Vertex')
plt.ylabel('Mean EMD Loss')
plt.title('Mean EMD Loss vs. Distance to Closest Convex Hull Vertex')
plt.legend()
plt.show()
