import wandb
import pandas as pd

api = wandb.Api()

runs = api.runs("yourname/cnn_sweep_project")

rows = []

for r in runs:
    cfg = r.config
    cfg["best_val_acc"] = r.summary.get("best_val_acc")
    rows.append(cfg)

df = pd.DataFrame(rows)
df.to_csv("experiments/wandb_full_results.csv", index=False)
