from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ExperimentConfig:
    # Identity
    name: str
    seed: int = 42

    # Data
    batch_size: int = 128
    image_size: int = 224
    num_classes: int = 8

    # Model
    backbone: str = "squeezenet1_0"
    feature_extraction: bool = True
    unfreeze_blocks: int = 0  # 0 is fully frozen
    dropout: bool = False
    dropout_prob: float = 0.5
    batch_norm: bool = False

    # Optimization
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9

    # Training
    epochs: int = 100
    early_stopping_patience: int = 20

    # Augmentation
    augmentation: Optional[str] = None  # "flip", "rotation", "color", etc.
