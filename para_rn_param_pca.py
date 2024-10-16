import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import random

# Set random seed for reproducibility
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LoRA(nn.Module):
    def __init__(self, input_size, output_size, rank=8, alpha=32):
        super(LoRA, self).__init__()
        self.rank = rank
        self.alpha = alpha

        # Original weight matrix (frozen during training)
        self.original_weight = nn.Parameter(torch.randn(output_size, input_size), requires_grad=False)

        # Low-rank matrices (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, input_size))
        self.lora_B = nn.Parameter(torch.randn(output_size, rank))

        # Scale the low-rank update
        self.scaling = alpha / rank

    def forward(self, x):
        # Original forward pass plus the LoRA update
        return torch.matmul(x, self.original_weight.T) + self.scaling * torch.matmul(x, torch.matmul(self.lora_A.T, self.lora_B.T))

# Define MLP with LoRA
class LoRA_MLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, lora_rank=8):
        super(LoRA_MLP, self).__init__()
        # First layer with LoRA
        self.fc1 = LoRA(input_size, hidden_size, rank=lora_rank)
        # Second layer without LoRA
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define a simple 2-layer MLP with a configurable hidden layer size
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def show_pca_variance(rn_features, n_components=3):
    # Initialize PCA
    pca = PCA(n_components=n_components)
    # Fit PCA on the data and transform the features
    reduced_features = pca.fit_transform(rn_features)
    print("Explained variance ratio:", sum(pca.explained_variance_ratio_))
    return reduced_features

def train_and_evaluate_mlp_for_user(X, Y, input_size, hidden_size=64, lora_rank=8):
    # Initialize the MLP model with the specified hidden size, loss function, and optimizer
    criterion = nn.MSELoss()
    # model = MLP(input_size, hidden_size)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model = LoRA_MLP(input_size, hidden_size, lora_rank)   
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=1e-2)

    # Training the model with tqdm for progress display
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Evaluate the model on the same training data
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        mse = mean_squared_error(y, predictions)
    
    # Concatenate all parameters into a single vector
    model_params_vector = torch.cat([param.view(-1) for param in model.parameters()])
    
    return model, mse, model_params_vector

def save_all_models_and_compare(user_models, save_path='user_models.pkl'):
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            saved_data = pickle.load(f)
            saved_models = saved_data['models']
            
            # Check differences between current models and saved models
            for user_id, new_model_data in user_models.items():
                if user_id in saved_models:
                    old_params_vector = saved_models[user_id]['params_vector']
                    new_params_vector = new_model_data['params_vector']
                    difference = np.linalg.norm(new_params_vector - old_params_vector)
                    print(f"Difference in params for user {user_id}: {difference:.6f}")
                else:
                    print(f"No existing model found for user {user_id}, saving new model.")
    else:
        print("No previous models found, saving all current models.")
    
    # Save the current models
    with open(save_path, 'wb') as f_out:
        pickle.dump({'models': user_models}, f_out)
    print(f"Models saved to {save_path}")

def plot_pca_2d_by_attribute(pca_results, attributes_df, attribute_name, title='2D PCA of MLP Parameters'):
    # Plot the 2D PCA results
    plt.figure(figsize=(10, 8))
    
    unique_values = attributes_df[attribute_name].unique()
    palette = sns.color_palette("hsv", len(unique_values))

    for i, value in enumerate(unique_values):
        indices = attributes_df[attribute_name] == value
        plt.scatter(pca_results[indices, 0], pca_results[indices, 1], color=palette[i], label=value)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f"{title} - {attribute_name}")
    plt.xlim([-3, 4.5])
    plt.ylim([-3, 4.5])        
    plt.legend()
    plt.savefig(f'{attribute_name}_MLP_PCA.jpg')

def plot_pca_2d_by_attribute_sep(pca_results, attributes_df, attribute_name, title='2D PCA of MLP Parameters'):
    unique_values = attributes_df[attribute_name].unique()
    n_values = len(unique_values)
    
    # Determine the layout for subplots based on the number of unique values
    n_cols = 3  # You can adjust the number of columns
    n_rows = (n_values + n_cols - 1) // n_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()  # Flatten the axes array to easily iterate over it
    
    palette = sns.color_palette("hsv", n_values)

    for i, value in enumerate(unique_values):
        indices = attributes_df[attribute_name] == value
        ax = axes[i]
        ax.scatter(pca_results[indices, 0], pca_results[indices, 1], color=palette[i], label=value)
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title(f"{title} - {attribute_name}: {value}")
        ax.set_xlim([-3, 4.5])
        ax.set_ylim([-3, 4.5])
        ax.legend()
    
    # If there are any unused subplots, turn them off
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f'{attribute_name}_MLP_PCA_subplots.jpg')


def load_testdata():
    # set_random_seed(seed=42)
    # Load the features from the pickle file
    with open('train_extracted_features.pkl', 'rb') as f:
        feature = pickle.load(f)

    # Convert the dictionary values to a stacked NumPy array
    rn_features = np.stack(list(feature.values()))
    img_path = feature.keys()
    
    # Set the number of PCA components
    n_components = 2048
    reduced_features = rn_features

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
    return final_merged_df, pca_columns

def representative_users(final_merged_df, number_users=40):
    # Step 1: Compute mean aesthetic scores for each image
    mean_scores = final_merged_df.groupby('imageName')['aestheticScore'].mean().reset_index()
    mean_scores.rename(columns={'aestheticScore': 'meanScore'}, inplace=True)

    # Step 2: Merge the mean scores with the original dataframe
    df = final_merged_df.merge(mean_scores, on='imageName')

    # Step 3: Compute the difference between the mean score and the personal score
    df['scoreDifference'] = df['aestheticScore'] - df['meanScore']

    # Step 4: Compute the MSE for each user, averaging out across images
    user_mse = df.groupby('userId')['scoreDifference'].apply(lambda x: np.mean(x**2)).reset_index()
    user_mse.rename(columns={'scoreDifference': 'MSE'}, inplace=True)

    # Step 5: Get the top 40 users with the smallest MSE
    top_n_users = user_mse.nsmallest(number_users, 'MSE')
    return top_n_users


if __name__ == '__main__':
    final_merged_df, pca_columns = load_testdata()
    print(representative_users(final_merged_df, 40))

    # Set a hidden layer size
    n_components = 2048
    hidden_size = 8

    # Prepare to store parameters and attributes
    user_params = []
    user_attributes = []
    user_models = {}

    # Training a 2-layer MLP for each userId without splitting the data
    for user_id in tqdm(final_merged_df['userId'].unique()):
        user_data = final_merged_df[final_merged_df['userId'] == user_id]
        # Use the entire data for both training and testing
        x = torch.tensor(user_data[pca_columns].values, dtype=torch.float32)
        y = torch.tensor(user_data['aestheticScore'].values, dtype=torch.float32).unsqueeze(1)

        # Train the MLP for this user and get the model parameters
        best_mse = float('inf')
        best_params_vector = None
        
        for _ in range(1):

            # Train the MLP for this user and get the model parameters
            model, mse, params_vector = train_and_evaluate_mlp_for_user(x, y, n_components, hidden_size)
            
            # Check if the current MSE is better than the best recorded one
            if mse < best_mse:
                best_mse = mse
                best_params_vector = params_vector
        
        tqdm.write(f'MSE for user {user_id}: {best_mse:.4f}')
        
        # Store the parameters and corresponding attributes
        user_params.append(best_params_vector.detach().numpy())
        user_attributes.append(user_data.iloc[0][['age', 'gender', 'EducationalLevel', 'artExperience', 'photographyExperience']].to_dict())

        # Save the best model for the user in memory
        user_models[user_id] = {
            'model': model.state_dict(),
            'mse': best_mse,
            'params_vector': best_params_vector.detach().numpy()
        }
    
    # Save all models and compare with previous saved models
    save_all_models_and_compare(user_models)
    
    # Convert to DataFrame for plotting
    user_params = np.vstack(user_params)
    attributes_df = pd.DataFrame(user_attributes)
    # Perform PCA to reduce the parameter vectors to 2D
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(np.array(user_params, dtype=float))

    # Plot the 2D PCA of the concatenated parameter vectors for each attribute
    for attr in ['age', 'gender', 'EducationalLevel', 'artExperience', 'photographyExperience']:
        plot_pca_2d_by_attribute(pca_results, attributes_df, attr)
        plot_pca_2d_by_attribute_sep(pca_results, attributes_df, attr)
    
    print("Plotting completed.")
