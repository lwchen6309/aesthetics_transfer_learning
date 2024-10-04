import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a dataframe
df = pd.read_csv('score2attr.csv')

# Set up the figure
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

# For each quality score, create a scatter plot
for idx, quality_score in enumerate(['GIAA', 'PIAA', 'PIAA user']):
    subset_score2attr = df[(df['QualityScore'] == quality_score) & (df['SROCC'] == 'Score2Attr')]
    subset_attr2score = df[(df['QualityScore'] == quality_score) & (df['SROCC'] == 'Attr2Score')]
    
    x_values = subset_score2attr.values[0][2:]
    y_values = subset_attr2score.values[0][2:]
    
    # Plotting
    axs[idx].scatter(x_values, y_values, facecolors='none', edgecolors='b', s=100, label=quality_score)
    
    # Adding text labels for each attribute
    for i, txt in enumerate(df.columns[1:-1]):
        axs[idx].annotate(txt, (x_values[i], y_values[i]), fontsize=8, ha='left')
    
    # Setting titles, labels and other aesthetics
    axs[idx].set_title(f'{quality_score}')
    axs[idx].set_xlabel('Score2Attr')
    axs[idx].set_ylabel('Attr2Score')
    # axs[idx].plot([0, 1], [0, 1], 'r--')  # reference line for comparison
    axs[idx].grid(True)
    
plt.tight_layout()
plt.show()