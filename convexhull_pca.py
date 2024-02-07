import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial import ConvexHull
import argparse
import os
import pickle
import random


def plot_pca_by_distance(X_pca, distances, title, filename):
    """
    Plot PCA-transformed data colored by distances to the center in original space.
    :param X_pca: PCA-transformed data.
    :param distances: Distances of each point in the original space to the center.
    :param title: Title of the plot.
    :param filename: Filename to save the plot.
    """
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=distances, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Distance to Center')
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(filename)
    # plt.show()



def plot_pca_data(X, labels, title, filename, label_dict):
    """
    Plot PCA-transformed data with different categories.
    :param X: PCA-transformed data.
    :param labels: List of labels for each data point.
    :param title: Title of the plot.
    :param filename: Filename to save the plot.
    :param label_dict: Dictionary to convert label indices to actual labels.
    """
    plt.figure(figsize=(6, 6))

    for label_idx in np.unique(labels):
        idx = labels == label_idx
        label = label_dict.get(label_idx, 'Unknown')
        plt.scatter(X[idx, 0], X[idx, 1], label=label, alpha=0.5, s=10)

    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.savefig(filename)


# Function to extract the outer shell of PCA-transformed data
def extract_outer_shell(X_transformed):
    # Calculate the convex hull of the points
    hull = ConvexHull(X_transformed)
    
    # Get the indices of the points on the convex hull
    indices_on_hull = hull.vertices
    
    # Select the points on the convex hull
    outer_shell_points = X_transformed[indices_on_hull]
    
    return outer_shell_points, indices_on_hull


age_dict = {'30-34': 0, '18-21': 1, '22-25': 2, '26-29': 3, '35-40': 4}
gender_dict = {'female': 0, 'male': 1}
education_dict = {'junior_college': 0, 'university': 1, 'senior_high_school': 2, 'technical_secondary_school': 3, 'junior_high_school': 4}
art_experience_dict = {'proficient': 0, 'competent': 1, 'beginner': 2, 'expert': 3}
photo_experience_dict = {'proficient': 0, 'competent': 1, 'beginner': 2, 'expert': 3}

# Reverse the dictionaries
age_dict_rev = {v: k for k, v in age_dict.items()}
gender_dict_rev = {v: k for k, v in gender_dict.items()}
education_dict_rev = {v: k for k, v in education_dict.items()}
art_experience_dict_rev = {v: k for k, v in art_experience_dict.items()}
photo_experience_dict_rev = {v: k for k, v in photo_experience_dict.items()}


if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Process PCA or PaCMAP on data.')
    parser.add_argument('--method', type=str, default='pac', choices=['random', 'pca'],
                        help='Method for dimensionality reduction: pacmap or pca (default: pacmap)')
    parser.add_argument('--num_user', type=int, default=50,
                        help='Number of users to consider (default: 200)')
    parser.add_argument('--is_reverse', action='store_true',
                        help='Whether to reverse the order of closest points (default: True)')
    parser.add_argument('--isplot', action='store_true',
                        help='Whether to plot the 2d data')   
    parser.add_argument('--convex_hull', action='store_true',
                        help='Whether to plot the 2d data')       
    parser.add_argument('--n_components', type=int, default=2,
                        help='Number of components for PCA/PaCMAP (default: 2)')

    
    # Parse the arguments
    args = parser.parse_args()

    method = args.method
    num_user = args.num_user
    is_reverse = args.is_reverse
    n_components = args.n_components

    # Load your data (make sure to exclude any non-feature columns like 'userId')
    # data = pd.read_csv('OneHotEncoded_Traits_Corrected.csv')
    file_path = 'precomputed_trait_encodings.pkl'
    with open(file_path, 'rb') as file:
        # Load the data using pickle
        precomputed_data = pickle.load(file)
    with open('users_list_more_than500imgs.txt', 'r') as file:
        user_ids_to_keep = [line.strip() for line in file]
    
    users_traits = []
    for user_id in user_ids_to_keep:
        traits = precomputed_data[user_id]['userTraits']
        output = [
            np.array(traits['age']),
            np.array(traits['gender']),
            np.array(traits['EducationalLevel']),
            np.array(traits['artExperience']),
            np.array(traits['photographyExperience']),
            np.array(traits['personality-E-onehot']),
            np.array(traits['personality-A-onehot']),
            np.array(traits['personality-N-onehot']),
            np.array(traits['personality-O-onehot']),
            np.array(traits['personality-C-onehot']),
            # np.array(traits['personality-E']),
            # np.array(traits['personality-A']),
            # np.array(traits['personality-N']),
            # np.array(traits['personality-O']),
            # np.array(traits['personality-C']),            
        ]
        output = np.concatenate(output)
        users_traits.append(output)
    
    user_ids = np.array(user_ids_to_keep)

    if method == 'random':
        # Save the unique closest user IDs to a CSV file
        unique_closest_user_ids = random.choices(user_ids, k=num_user)
        unique_closest_user_ids_df = pd.DataFrame(unique_closest_user_ids, columns=['userId'])
        for i in range(10):
            filename = 'shell_%duser_ids_%s_%d.csv'%(num_user, method, i)
            filename = os.path.join('shell_users', 'random_users', filename)
            unique_closest_user_ids_df.to_csv(filename, index=False)
        raise Exception('Pause by random user')

    iaa_data = np.stack(users_traits)
    # iaa_data[:, iaa_data.std(0) > 0]

    # Calculate the center of the original data (mean in each dimension)
    center = np.mean(iaa_data, axis=0)
    center_distances = np.linalg.norm(iaa_data - center, axis=1)

    # Fit and transform the data using PCA
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(iaa_data)

    if args.convex_hull:
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
    else:
        closest_center_indices = np.argsort(center_distances)
        closest_vertex_indices = np.flip(closest_center_indices)
        if is_reverse:
            closest_vertex_indices = np.flip(closest_vertex_indices)

    # Select the corresponding user IDs and data points for the unique closest vertices
    closest_vertex_indices = closest_vertex_indices[:num_user]
    unique_closest_user_ids = user_ids[closest_vertex_indices]
    unique_closest_data_points = X_transformed[closest_vertex_indices]

    # Save the unique closest user IDs to a CSV file
    unique_closest_user_ids_df = pd.DataFrame(unique_closest_user_ids, columns=['userId'])
    filename = '%dD_shell_%duser_ids_%s.csv'%(n_components, num_user, method)
    if is_reverse:
        filename = filename.replace('.csv', '_rev.csv')
    filename = os.path.join('shell_users', '500imgs', filename)
    unique_closest_user_ids_df.to_csv(filename, index=False)

    if args.isplot and n_components == 2:
        # Plot for different categories like gender, age, etc.
        traits_dictionaries = {
            'age': age_dict_rev,
            'gender': gender_dict_rev,
            'EducationalLevel': education_dict_rev,
            'artExperience': art_experience_dict_rev,
            'photographyExperience': photo_experience_dict_rev
        }
        for trait, trait_dict in traits_dictionaries.items():
            trait_labels = np.array([np.argmax(precomputed_data[user_id]['userTraits'][trait]) for user_id in user_ids_to_keep])
            plot_pca_data(X_transformed, trait_labels, f'PCA Plot with {trait}', f'PCA_{trait}.png', trait_dict)

        transformed_center = np.mean(X_transformed, axis=0)
        tranformed_center_distances = np.linalg.norm(X_transformed - transformed_center, axis=1)    
        plt.figure(figsize=(6, 6))
        plt.scatter(tranformed_center_distances, center_distances)
        plt.xlabel('PCA Distance to PCA Center')
        plt.ylabel('Distnce to Center')
        plot_pca_by_distance(X_transformed, center_distances, 'PCA Plot Colored by Distance', 'Distances.jpg')
        plot_pca_by_distance(X_transformed, tranformed_center_distances, 'PCA Plot Colored by PCA Distance', 'PCA_Distances.jpg')

        # Plot the extracted outer shell
        if args.convex_hull:
            plt.figure(figsize=(6, 6))
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=1)
            plt.scatter(outer_shell[:, 0], outer_shell[:, 1], s=10, alpha=0.5, color='red', label='Outer Shell')
            plt.title('%s Embedding with Outer Shell'%method)
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
        
        # Visualize the 100 unique data points
        plt.figure(figsize=(6, 6))
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], s=1, label='All Users')
        # plt.scatter(np.mean(X_transformed[:, 0]), np.mean(X_transformed[:, 1]), s=40, label='Center')
        plt.scatter(unique_closest_data_points[:, 0], unique_closest_data_points[:, 1], s=10, alpha=0.5, color='red', label='Selected Users')
        plt.title('Visualization of Selected Users')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()

        plt.show()