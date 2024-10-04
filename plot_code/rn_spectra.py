import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.timeseries import LombScargle
import pandas as pd
import os


def show_pca_variance(rn_features, n_features=2):
    # Initialize PCA, using min(n_samples, n_features) components
    pca = PCA(n_features)

    # Fit PCA on the data
    pca.fit(rn_features)

    print("Explained variance ratio:", pca.explained_variance_ratio_[:10])

    # Plotting the singular values
    plt.figure(figsize=(10, 6))
    plt.plot(pca.singular_values_, marker='o')
    plt.title('Singular Values from PCA')
    plt.xlabel('Component number')
    plt.ylabel('Singular value')
    plt.grid(True)
    plt.savefig('RN_PCA.jpg')
    # plt.show()


# Load the features from the pickle file
with open('train_extracted_features.pkl', 'rb') as f:
# with open('test_extracted_features.pkl', 'rb') as f:
    feature = pickle.load(f)

# Convert the dictionary values to a stacked NumPy array
rn_features = np.stack(list(feature.values()))
img_path = [os.path.basename(x) for x in feature.keys()]

# show_pca_variance(rn_features)

# Initialize PCA with 2 components
pca = PCA(n_components=2)

# Fit PCA on the data and transform the features
reduced_features = pca.fit_transform(rn_features)
pd_table = pd.DataFrame(reduced_features, columns=['PCA1', 'PCA2'])
pd_table['imageName'] = img_path

# Load the CSV files
subdir = "~/datasets/PARA/annotation/"
images_df = pd.read_csv(os.path.join(subdir, "PARA-Images.csv"))
user_info_df = pd.read_csv(os.path.join(subdir, "PARA-UserInfo.csv"))

# Merging the datasets on 'userId'
merged_df = pd.merge(images_df, user_info_df, on='userId', how='left')

# Merge pd_table with merged_df using 'imageName' as the key
final_merged_df = pd.merge(merged_df, pd_table, on='imageName', how='inner')

# Now final_merged_df contains all the data merged together
print(final_merged_df.head())

# pca = final_merged_df[['PCA1', 'PCA2']].values
# big5 = final_merged_df[['personality-E', 'personality-A', 'personality-N', 'personality-O', 'personality-C']].values.astype(np.float64) / 10.

users = final_merged_df['userId'].unique()
std_scores = []
for user in users:
    u_df = final_merged_df[final_merged_df['userId'] == user]
    # pca = u_df[['PCA1', 'PCA2']].values
    # big5 = u_df[['personality-E', 'personality-A', 'personality-N', 'personality-O', 'personality-C']].values.astype(np.float64) / 10.
    std_scores.append(u_df['aestheticScore'].std())

# Plot histogram of standard deviations
plt.figure()
plt.hist(np.array(std_scores), bins=20, alpha=0.75)
plt.xlabel('Standard Deviation of Aesthetic Scores')
plt.ylabel('Frequency')
plt.title('Histogram of Standard Deviations of Aesthetic Scores by User')
plt.grid(True)
plt.savefig('tmp.jpg')

# Extract the PCA1 and aestheticScore columns
# x = pca.mean(1)
# y = big5.mean(1)
# z = final_merged_df['aestheticScore'].values
# # Create a figure for the plot
# fig = plt.figure(figsize=(14, 6))

# # 3D scatter plot
# ax1 = fig.add_subplot(111, projection='3d')
# scatter = ax1.scatter(x, y, z, c=z, cmap='viridis', marker='o')

# # Add labels and title
# ax1.set_xlabel('Imgae')
# ax1.set_ylabel('Big5')
# ax1.set_zlabel('Scores')
# ax1.set_title('3D Scatter Plot of Imgae, Big5, and Scores')

# # Adding a color bar to indicate the values of z
# plt.colorbar(scatter, ax=ax1, label='Scores')
# plt.savefig('3D_aesthetic.jpg')
