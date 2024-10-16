import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import pickle
import os
SUBDIR = './figure_outs'


def show_pca_variance(rn_features, n_components=1):
    # Initialize PCA
    pca = PCA(n_components=n_components)
    # Fit PCA on the data and transform the features
    reduced_features = pca.fit_transform(rn_features)
    print("Explained variance ratio:", sum(pca.explained_variance_ratio_))
    return reduced_features


def plot_pca_2d_with_smoothed_score(df, attribute_name, pca_columns, title='2D PCA Features with Aesthetic Score'):
    # Create a 2D plot to show img_features and smoothed mean score (mu)
    plt.figure(figsize=(10, 8))

    unique_values = df[attribute_name].unique()
    palette = sns.color_palette("tab10", len(unique_values))

    # Pre-define a plot dictionary to accumulate sorted points for each unique value
    plot_data = {value: {'x': [], 'y': [], 'color': palette[i]} for i, value in enumerate(unique_values)}

    # Collect overall mean scores and img features for all images
    all_data = []
    for img, group in df.groupby('imageName'):
        img_features = group[pca_columns].values[0]
        mu = group['aestheticScore'].mean()
        all_data.append([img_features[0], mu])

    # Convert collected data into a numpy array for easier processing
    all_data = np.array(all_data)
    all_data_sorted = all_data[all_data[:, 0].argsort()]
    sorted_overall_x = all_data_sorted[:, 0]
    sorted_overall_y = all_data_sorted[:, 1]
    sorted_overall_y = gaussian_filter1d(sorted_overall_y, sigma=50)  # Adjust sigma for smoother/less smooth


    # Group by 'attribute_name' and 'imageName' to collect data for each unique group
    grouped = df.groupby([attribute_name, 'imageName'])
    
    for (value, img), group in grouped:
        img_features = group[pca_columns].values[0]
        mu = group['aestheticScore'].mean()

        # Accumulate data for each unique value
        plot_data[value]['x'].append(img_features[0])
        plot_data[value]['y'].append(mu)

    # Sort and plot smoothed lines for each unique value
    for value, data in plot_data.items():
        # Sort data based on img_features (PCA Feature 1)
        sorted_indices = sorted(range(len(data['x'])), key=lambda k: data['x'][k])
        sorted_x = [data['x'][i] for i in sorted_indices]
        sorted_y = [data['y'][i] for i in sorted_indices]

        # Apply Gaussian kernel smoothing to y-values
        y_smoothed = gaussian_filter1d(sorted_y, sigma=50)  # Adjust sigma for smoother/less smooth

        # Plot the smoothed line for each unique value
        plt.plot(sorted_x, y_smoothed, color=data['color'], label=f'{value}')

    # Plot the overall mean scores as a distinct line
    y_smoothed = gaussian_filter1d(sorted_y, sigma=50)
    plt.plot(sorted_overall_x, sorted_overall_y, color='black', linestyle='--', linewidth=2, label='Overall Mean')

    plt.xlabel('PCA Feature 1')
    plt.ylabel('Mean Aesthetic Score (μ)')
    plt.title(title)

    # Add legend
    plt.legend(title=attribute_name)

    # Save the figure to the global SUBDIR path
    plt.savefig(os.path.join(SUBDIR, f'{attribute_name}_PCA_with_Kernel_Smoothed_Score_and_Overall_Mean_2D.jpg'))


if __name__ == '__main__':
    # Load the features from the pickle file
    # with open('train_extracted_features.pkl', 'rb') as f:
    with open('test_extracted_features.pkl', 'rb') as f:
        feature = pickle.load(f)
    
    # Convert the dictionary values to a stacked NumPy array
    rn_features = np.stack(list(feature.values()))
    img_path = list(feature.keys())
    
    cosim = rn_features@rn_features.T
    sorted_indices = np.argsort(cosim[0])[::-1]
    img_path = [img_path[idx] for idx in sorted_indices]
    cosim_permuted = cosim[np.ix_(sorted_indices, sorted_indices)]
    # plt.imshow(cosim_permuted)
    # plt.savefig(os.path.join(SUBDIR,'cosim.jpg'))
    # raise Exception
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
    attrs = ['age', 'gender', 'EducationalLevel', 'artExperience', 'photographyExperience']
    for attr in attrs:
        plot_pca_2d_with_smoothed_score(final_merged_df, attr, pca_columns)
    print("Plotting completed.")
