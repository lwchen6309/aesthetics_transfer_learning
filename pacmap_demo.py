import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import pacmap
import numpy as np
from scipy.spatial import ConvexHull
import argparse
import os


def plot_pca_by_onehot_trait(X_transformed, data, trait_prefix, method):
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
    
    plt.title(f'%s Embedding Colored by {trait_prefix}'%method)
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


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Process PCA or PaCMAP on data.')
    parser.add_argument('--method', type=str, default='pacmap', choices=['pacmap', 'pca'],
                        help='Method for dimensionality reduction: pacmap or pca (default: pacmap)')
    parser.add_argument('--num_user', type=int, default=50,
                        help='Number of users to consider (default: 200)')
    parser.add_argument('--is_reverse', action='store_true',
                        help='Whether to reverse the order of closest points (default: True)')
    parser.add_argument('--isplot', action='store_true',
                        help='Whether to plot the 2d data')   
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for PCA/PaCMAP (default: 2)')

    
    # Parse the arguments
    args = parser.parse_args()

    method = args.method
    num_user = args.num_user
    is_reverse = args.is_reverse
    isplot = args.isplot
    n_components = args.n_components


    # Load your data (make sure to exclude any non-feature columns like 'userId')
    data = pd.read_csv('OneHotEncoded_Traits_Corrected.csv')
    X = data.drop('userId', axis=1)
    
    # Initialize PCA instance for 2 components
    if method == 'pacmap':    
        embedding = pacmap.PaCMAP(n_components=n_components, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
        X_transformed = embedding.fit_transform(X, init="pca")
    else:
        # Fit and transform the data using PCA
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X)

    # Extract the outer shell of the PCA-transformed data and the shell indices
    print('Compute Convex hull')
    outer_shell, shell_indices = extract_outer_shell(X_transformed)
    print('Finish Convex hull')
    # Calculate distances between each data point and all shell vertices
    distances = np.linalg.norm(X_transformed[:, np.newaxis] - outer_shell, axis=2)

    # Find the indices of the closest shell vertex for each data point
    closest_vertex_distance = np.min(distances, axis=1)
    closest_vertex_indices = np.argsort(closest_vertex_distance)
    if is_reverse:
        closest_vertex_indices = np.flip(closest_vertex_indices)
    closest_vertex_indices = closest_vertex_indices[:num_user]
    
    # Select the corresponding user IDs and data points for the unique closest vertices
    unique_closest_user_ids = data['userId'][closest_vertex_indices]
    unique_closest_data_points = X_transformed[closest_vertex_indices]

    # Save the unique closest user IDs to a CSV file
    unique_closest_user_ids_df = pd.DataFrame(unique_closest_user_ids, columns=['userId'])
    filename = '%dD_shell_%duser_ids_%s.csv'%(n_components, num_user, method)
    if is_reverse:
        filename = filename.replace('.csv', '_rev.csv')
    filename = os.path.join('shell_users', filename)
    unique_closest_user_ids_df.to_csv(filename, index=False)

    if isplot and n_components == 2:
        # Plot data for each class
        plot_pca_by_onehot_trait(X_transformed, data, 'gender_', method)
        plot_pca_by_onehot_trait(X_transformed, data, 'age_', method)
        plot_pca_by_onehot_trait(X_transformed, data, 'EducationalLevel_', method)
        plot_pca_by_onehot_trait(X_transformed, data, 'artExperience_', method)
        plot_pca_by_onehot_trait(X_transformed, data, 'photographyExperience_', method)

        # Plot the extracted outer shell
        plt.figure(figsize=(6, 6))
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=0.6)
        plt.scatter(outer_shell[:, 0], outer_shell[:, 1], s=10, color='red', label='Outer Shell')
        plt.title('%s Embedding with Outer Shell'%method)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()

        # Visualize the 100 unique data points
        plt.figure(figsize=(8, 8))
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=0.6, label='All Data')
        plt.scatter(unique_closest_data_points[:, 0], unique_closest_data_points[:, 1], s=10, color='red', label='Unique Closest Data')
        plt.title('Visualization of Unique Closest Data Points')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

        plt.show()