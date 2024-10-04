import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Generating two random Gaussian distributions
    data1 = np.random.normal(loc=0.5, scale=0.2, size=1000)
    data2 = np.random.normal(loc=0.9, scale=0.2, size=1000)
    data_sum = np.concatenate([data1, data2])

    # Creating subplots
    fig, axs = plt.subplots(3, 1, figsize=(5, 10), sharex=True)

    # Plotting the first Gaussian distribution
    axs[0].hist(data1, bins=10, color='blue', alpha=0.7)
    axs[0].set_title('Score Distribution for male')

    # Plotting the second Gaussian distribution
    axs[1].hist(data2, bins=10, color='green', alpha=0.7)
    axs[1].set_title('Score Distribution for female')

    # Plotting the summation of the two Gaussian distributions
    axs[2].hist(data_sum, bins=10, color='purple', alpha=0.7)
    axs[2].set_title('Sumed Score Distribution')

    # Displaying the plots
    plt.tight_layout()
    plt.show()