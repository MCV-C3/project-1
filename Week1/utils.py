import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_imgs_at_idx(target_index,data_test,methods,data):
    classes = [
'shopping and dining', 'sports and leisure', 'water_ice_snow', 'mountains_hills_desert_sky', 'workplace', 'home or hotel', 'forest_field_jungle', 'industrial and construction', 'houses_cabins_gardens_farms', 'commercial buildings', 'sports_fields'

    ]

    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [1, 1]})
    # --- Image Plot ---
    ax_img.imshow(data_test[target_index][0], cmap='gray')
    ax_img.set_title(f"Image at Index {target_index}", fontsize=12)
    ax_img.axis('off')

    # --- Text Plot ---
    ax_text.axis('off') # Hide axes for the text area
    ax_text.set_title("Mismatch Predictions", fontsize=12)

    text_y = 0.9
    line_height = 0.15

    # Iterate through the indices where the secondary prediction was FALSE
    for method in methods:
        predicted_class = classes[data[method][0][target_index]]
        
        # Format the text string
        text_line = f"Method: {method}\nPrediction: {predicted_class}"
        if data[method][0][target_index] == data[method][1][target_index]:
            # Add the text in green
            ax_text.text(0.1, text_y, text_line, color='green', fontsize=11, verticalalignment='top')
        else:
            ax_text.text(0.1, text_y, text_line, color='red', fontsize=11, verticalalignment='top')
        text_y -= line_height


    plt.tight_layout()
    plt.show()

def box_plot(data):
    # Create a copy to avoid SettingWithCopyWarning
    data = data.copy()
    
    # Set style similar to the reference images
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    # Create figure with larger size for better visibility
    fig, ax = plt.subplots(figsize=(12, 7))
    # Assuming you have your scaling dataframe with CV_Accuracy and CV_Std
    # If you want violin plots (like Image 1), you'd need the full distribution data
    # For now, I'll create a hybrid using box plots with your existing data
    data['Value'] = data['Value'].astype(str)
    # Create positions for each scaling method
    positions = np.arange(len(data))
    # Create box plot elements manually to match the style
    for i, (idx, row) in enumerate(data.iterrows()):
        # Simulate distribution from mean and std
        # In practice, you'd use your actual CV fold results here
        mean = row['CV_Accuracy']
        std = row['CV_Std']
        
        # Create a box plot style visualization
        # Box represents mean ± 0.5*std, whiskers show mean ± std
        box_bottom = mean - 0.5 * std
        box_top = mean + 0.5 * std
        whisker_bottom = max(0, mean - std)
        whisker_top = min(1, mean + std)
        
        # Choose color from viridis palette
        color = sns.color_palette("viridis", len(data))[i]
        
        # Draw whiskers
        ax.plot([i, i], [whisker_bottom, whisker_top], 'k-', linewidth=1.5)
        
        # Draw box
        box = plt.Rectangle((i - 0.3, box_bottom), 0.6, box_top - box_bottom,
                            facecolor=color, edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.add_patch(box)
        
        # Draw median line (at mean)
        ax.plot([i - 0.3, i + 0.3], [mean, mean], 'k-', linewidth=2)
        
        # Add caps to whiskers
        ax.plot([i - 0.15, i + 0.15], [whisker_bottom, whisker_bottom], 'k-', linewidth=1.5)
        ax.plot([i - 0.15, i + 0.15], [whisker_top, whisker_top], 'k-', linewidth=1.5)
    # Customize the plot
    ax.set_xticks(positions)
    ax.set_xticklabels(data['Value'], fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Method', fontsize=18, fontweight='bold')
    ax.set_title('CV Accuracy', 
                fontsize=20, fontweight='bold', pad=20)
    ax.tick_params(axis='y', labelsize=14)
    # Set y-axis limits and format as percentage
    ax.set_ylim(0.2, 0.35)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}'))
    # Add grid for better readability (matching Image 2 style)
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()