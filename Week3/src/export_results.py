import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# CONFIGURE THESE
# -----------------------------
PROJECT = "project-1-Week3_src"
ENTITY  = "raim8218-maulana-abul-kalam-azad-university-of-technolog"
OUTPUT_DIR = "experiments"
PLOT_DIR = "plots"
CSV_NAME = "wandb_full_results.csv"

# -----------------------------
# SETUP
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("Fetching runs from Weights & Biases...")

api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}")

records = []

for r in runs:
    
    cfg = r.config
    
    row = {
        "run_id": r.id,
        "model": cfg.get("model"),
        "dropout": cfg.get("dropout"),
        "optimizer": cfg.get("optimizer"),
        "learning_rate": cfg.get("learning_rate"),
        "batch_size": cfg.get("batch_size"),
        "weight_decay": cfg.get("weight_decay"),
        "epochs": cfg.get("epochs"),
        "best_val_acc": r.summary.get("best_val_acc"),
        "final_val_acc": r.summary.get("val_acc"),
    }
    
    records.append(row)

df = pd.DataFrame(records)

# Save CSV
csv_path = os.path.join(OUTPUT_DIR, CSV_NAME)
df.to_csv(csv_path, index=False)
print(f"Saved CSV results to {csv_path}")

# ---------------------------------
#   PLOTTING HELPERS
# ---------------------------------
def save_scatter(x, y, title, xlabel, ylabel, filename):
    plt.figure()
    plt.scatter(df[x], df[y])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

def save_bar(x, y, title, xlabel, ylabel, filename):
    plt.figure()
    df.groupby(x)[y].mean().plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

# ---------------------------------
#   GENERATE PLOTS
# ---------------------------------

print("Generating plots...")

# Accuracy vs Learning rate
if "learning_rate" in df.columns:
    save_scatter(
        "learning_rate", 
        "best_val_acc",
        "Validation Accuracy vs Learning Rate",
        "Learning Rate",
        "Best Validation Accuracy",
        "acc_vs_lr.png"
    )

# Accuracy vs Dropout
if "dropout" in df.columns:
    save_scatter(
        "dropout",
        "best_val_acc",
        "Validation Accuracy vs Dropout",
        "Dropout Rate",
        "Best Validation Accuracy",
        "acc_vs_dropout.png"
    )

# Accuracy vs Batch Size
if "batch_size" in df.columns:
    save_scatter(
        "batch_size",
        "best_val_acc",
        "Validation Accuracy vs Batch Size",
        "Batch Size",
        "Best Validation Accuracy",
        "acc_vs_batchsize.png"
    )

# Accuracy vs Weight Decay
if "weight_decay" in df.columns:
    save_scatter(
        "weight_decay",
        "best_val_acc",
        "Validation Accuracy vs Weight Decay",
        "Weight Decay",
        "Best Validation Accuracy",
        "acc_vs_weightdecay.png"
    )

# Accuracy by Model Type
if "model" in df.columns:
    save_bar(
        "model",
        "best_val_acc",
        "Mean Validation Accuracy by Model Architecture",
        "Model",
        "Mean Best Validation Accuracy",
        "acc_by_model.png"
    )

# Accuracy by Optimizer
if "optimizer" in df.columns:
    save_bar(
        "optimizer",
        "best_val_acc",
        "Mean Validation Accuracy by Optimizer",
        "Optimizer",
        "Mean Best Validation Accuracy",
        "acc_by_optimizer.png"
    )

print("All plots saved in:", PLOT_DIR)
