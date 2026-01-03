import wandb
import os

os.system("wandb sweep src/sweep_config.yaml")
