import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def show_pca_variance(rn_features, n_components=1):
    # Initialize PCA
    pca = PCA(n_components=n_components)
    # Fit PCA on the data and transform the features
    reduced_features = pca.fit_transform(rn_features)
    print("Explained variance ratio:", sum(pca.explained_variance_ratio_))
    return reduced_features

def plot_pca_1d_with_score(pca_results, scores, attributes_df, attribute_name, title='1D PCA of Features with Aesthetic Score'):
    # Plot the 1D PCA results with aesthetic score as the y-axis
    plt.figure(figsize=(10, 8))
    
    unique_values = attributes_df[attribute_name].unique()
    palette = sns.color_palette("hsv", len(unique_values))
    
    for i, value in enumerate(unique_values):
        indices = attributes_df[attribute_name] == value
        plt.scatter(pca_results[indices], scores[indices], color=palette[i], label=value)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('Aesthetic Score')
    plt.title(f"{title} - {attribute_name}")
    plt.legend()
    plt.savefig(f'{attribute_name}_PCA_with_Score.jpg')
    plt.close()


if __name__ == '__main__':
    # Load the features from the pickle file
    with open('extracted_features.pkl', 'rb') as f:
        feature = pickle.load(f)

    # Convert the dictionary values to a stacked NumPy array
    rn_features = np.stack(list(feature.values()))
    img_path = list(feature.keys())
    
    cosim = rn_features@rn_features.T
    sorted_indices = np.argsort(cosim[0])[::-1]
    img_path = [img_path[idx] for idx in sorted_indices]
    cosim_permuted = cosim[np.ix_(sorted_indices, sorted_indices)]
    plt.imshow(cosim_permuted)
    plt.savefig('cosim.jpg') 
    raise Exception
    # Set the number of PCA components
    n_components = 1
    reduced_features = show_pca_variance(rn_features, n_components=n_components)

    # Create a DataFrame to store the PCA components
    pca_columns = [f'PCA{i+1}' for i in range(n_components)]
    pd_table = pd.DataFrame(reduced_features, columns=pca_columns)
    pd_table['imageName'] = [os.path.basename(path) for path in img_path]

    # Load the CSV files
    subdir = "/home/lwchen/datasets/PARA/annotation/"
    images_df = pd.read_csv(os.path.join(subdir, "PARA-Images.csv"))
    user_info_df = pd.read_csv(os.path.join(subdir, "PARA-UserInfo.csv"))

    # Merging the datasets on 'userId'
    merged_df = pd.merge(images_df, user_info_df, on='userId', how='left')

    # Merge pd_table with merged_df using 'imageName' as the key
    final_merged_df = pd.merge(merged_df, pd_table, on='imageName', how='inner')

    # Print the head of the final merged DataFrame
    print(final_merged_df.head())

    # Prepare to store attributes and scores
    user_attributes = final_merged_df[['age', 'gender', 'EducationalLevel', 'artExperience', 'photographyExperience']]
    scores = final_merged_df['aestheticScore'].values

    # Perform PCA to reduce the feature vectors to 1D
    pca = PCA(n_components=1)
    pca_results = pca.fit_transform(final_merged_df[pca_columns].values)

    # Plot the 1D PCA of the features with aesthetic scores
    for attr in ['age', 'gender', 'EducationalLevel', 'artExperience', 'photographyExperience']:
        plot_pca_1d_with_score(pca_results, scores, user_attributes, attr)
    
    print("Plotting completed.")
