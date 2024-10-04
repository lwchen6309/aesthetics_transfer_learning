import matplotlib.pyplot as plt
import numpy as np

# Sampling data from a normal distribution with mean = 4 and std = 3
data = [np.random.normal(2, 1, 1000), 
        np.random.normal(4, 3, 1000),
        np.random.normal(3, 0.001, 1000)
        ]

fig, axs = plt.subplots(1, len(data), figsize=(8, 6))  # Define the figure size here
for ax, dataset in zip(axs, data):
    data_clipped = np.clip(dataset, 0, 5)  # Clipping data to be within the 0-5 range
    ax.hist(data_clipped, bins=5, range=(1, 5), edgecolor='black')
    ax.set_xlabel('IAA Score')  # Setting x-axis labels
    ax.set_ylabel('')  # Hiding y-axis labels
    ax.tick_params(labelleft=False, labelbottom=True)  # Making y-axis tick labels invisible

plt.show()