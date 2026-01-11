import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from torchvision.datasets import ImageFolder


# 1. Setup paths and parameters (Adjust these to your local environment)
base_path =  "/home/msiau/data/tmp/jventosa/2425"
choosen_split = "1" 
transformation = None # Replace with your actual transforms.Compose([...])

# 2. Load the datasets
data_train = ImageFolder(f"{base_path}/MIT_small_train_{choosen_split}/train", transform=transformation)
data_test = ImageFolder(f"{base_path}/MIT_small_train_{choosen_split}/test", transform=transformation)

def check_balance(dataset, name="Dataset"):
    # Extract all labels from the dataset
    # dataset.targets is a list of integers corresponding to the class index
    counts = Counter(dataset.targets)
    
    # Map the indices back to class names for readability
    class_names = dataset.classes
    readable_counts = {class_names[i]: count for i, count in counts.items()}
    
    print(f"--- {name} Balance Report ---")
    for cls, count in readable_counts.items():
        percentage = (count / len(dataset)) * 100
        print(f"Class: {cls:15} | Count: {count:5} | ({percentage:.2f}%)")
    
    # Check if perfectly balanced
    if len(set(counts.values())) == 1:
        print(f"\n✅ The {name} is perfectly balanced.")
    else:
        print(f"\n⚠️ The {name} is imbalanced.")
    print("-" * 30)

# 3. Execute the check
check_balance(data_train, "Training Set")
check_balance(data_test, "Test Set")

def plot_class_ratios(data_train, data_test):
    # Get class names and counts
    classes = data_train.classes
    train_counts = Counter(data_train.targets)
    test_counts = Counter(data_test.targets)
    
    train_total = len(data_train)
    test_total = len(data_test)

    # Build a list of dictionaries for the DataFrame
    data_list = []
    for i, class_name in enumerate(classes):
        # Add training ratio
        data_list.append({
            'Class': class_name,
            'Ratio': train_counts[i] / train_total,
            'Set': 'Train'
        })
        # Add test ratio
        data_list.append({
            'Class': class_name,
            'Ratio': test_counts[i] / test_total,
            'Set': 'Test'
        })

    # Create DataFrame
    df = pd.DataFrame(data_list)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(data=df, x='Class', y='Ratio', hue='Set', palette='viridis')
    
    # Formatting
    plt.title('Dataset Distribution Train vs Test Set', fontsize=15)
    plt.ylabel('Ratio', fontsize=12)
    plt.xlabel('Class Name', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, max(df['Ratio']) * 1.2) # Give some space for labels
    
    # Optional: Add percentage labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=9)

    plt.tight_layout()
    plt.savefig('class_distribution_ratios.png')
    plt.show()

# Execute the function
plot_class_ratios(data_train, data_test)