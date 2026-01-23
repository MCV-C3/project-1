import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import List, Dict, Union, Type, Callable, Optional

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    GradCAMPlusPlus = None
    ClassifierOutputTarget = None


#Model blocks

class SEAttention(nn.Module):
    """Squeeze-and-Excitation Block (Channel Attention)"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv -> BN -> ReLU -> Dropout"""
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, 
                 use_bn=True, dropout_prob=0.0):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
        if dropout_prob > 0:
            layers.append(nn.Dropout2d(p=dropout_prob))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class InceptionBlock(nn.Module):
    """Simple Inception Module"""
    def __init__(self, in_c, out_c):
        super().__init__()
        branch_channels = out_c // 4
        
        # 1x1 conv branch
        self.b1 = nn.Conv2d(in_c, branch_channels, kernel_size=1)
        
        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_c, branch_channels, kernel_size=1),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1)
        )
        
        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_c, branch_channels, kernel_size=1),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2)
        )
        
        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_c, branch_channels, kernel_size=1)
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)

class ResidualWrapper(nn.Module):
    """Wraps any block to add a residual connection"""
    def __init__(self, block, in_c, out_c):
        super().__init__()
        self.block = block
        self.shortcut = nn.Identity()
        
        # If dimensions change, use 1x1 conv to match them
        if in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

# ==========================================
# 2. THE MODULAR MODEL WRAPPER
# ==========================================

class ModularCNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 3, config: List[Dict] = None):
        """
        Args:
            num_classes: Output dimension.
            input_channels: Channels in input image (usually 3).
            config: A list of dictionaries describing the layers.
                    Example: [{'type': 'conv', 'out': 32, 'k': 3, 'bn': True}, ...]
        """
        super().__init__()
        self.config = config
        
        # 1. Build Feature Extractor
        self.features, last_channel_count = self._build_features(input_channels, config)
        
        # 2. Build Classifier (Global Average Pooling + Linear)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channel_count, num_classes)

    def _build_features(self, current_channels, config):
        layers = []
        
        for layer_cfg in config:
            l_type = layer_cfg.get('type', 'conv')
            
            # Common params
            out_c = layer_cfg.get('out', current_channels)
            bn = layer_cfg.get('bn', False)
            drop = layer_cfg.get('dropout', 0.0)
            residual = layer_cfg.get('residual', False)

            block = None

            if l_type == 'conv':
                k = layer_cfg.get('k', 3)
                s = layer_cfg.get('s', 1)
                p = layer_cfg.get('p', 1)
                block = ConvBlock(current_channels, out_c, k, s, p, bn, drop)
            
            elif l_type == 'inception':
                block = InceptionBlock(current_channels, out_c)
                # Inception usually maintains or changes channels internally,
                # here we force output to be whatever is requested.
            
            elif l_type == 'maxpool':
                k = layer_cfg.get('k', 2)
                s = layer_cfg.get('s', 2)
                block = nn.MaxPool2d(kernel_size=k, stride=s)
                # MaxPool doesn't change channels
                out_c = current_channels 

            elif l_type == 'attention':
                # Adds an attention mechanism without changing geometry
                block = SEAttention(current_channels)
                out_c = current_channels

            # --- Wrap with Residual if requested ---
            if residual and l_type not in ['maxpool', 'attention']:
                block = ResidualWrapper(block, current_channels, out_c)

            if block:
                layers.append(block)
                current_channels = out_c

        return nn.Sequential(*layers), current_channels

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # just for analysis

    def extract_grad_cam(self, input_image: torch.Tensor, 
                         target_layer_idx: int = -1, 
                         target_category: int = None):
        """
        Extracts GradCAM heatmap.
        target_layer_idx: Index of layer in self.features to visualize (default: last conv layer)
        """
        if GradCAMPlusPlus is None:
            raise ImportError("pytorch_grad_cam not installed.")

        # Identify target layer. If -1, find the last Conv2d or Inception block
        if target_layer_idx == -1:
            target_layer = [self.features[-1]]
        else:
            target_layer = [self.features[target_layer_idx]]
            
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None

        with GradCAMPlusPlus(model=self, target_layers=target_layer) as cam:
            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]
            
        return grayscale_cam

    def extract_feature_maps(self, input_image: torch.Tensor):
        """Returns feature maps and names for all layers in feature extractor"""
        maps = []
        names = []
        x = input_image.clone()
        
        for idx, layer in enumerate(self.features):
            x = layer(x)
            # Only keep maps with spatial dimensions > 1
            if x.dim() == 4 and x.shape[2] > 1:
                maps.append(x)
                names.append(f"Layer_{idx}_{layer.__class__.__name__}")
        return maps, names


# ==========================================
# 4. EXPERIMENT HELPER (The "Parameter Wrapper")
# ==========================================

def create_experiment_model(
    num_classes: int,
    num_conv_layers: int = 3,
    base_channels: int = 32,
    use_bn: bool = False,
    use_dropout: bool = False,
    dropout_prob: float = 0.3,
    use_residual: bool = False,
    use_inception: bool = False,
    use_attention: bool = False
) -> ModularCNN:
    """
    Factory function to generate a model based on high-level experiment flags.
    This replaces hardcoding and enables your flowchart iteration.
    """
    
    config = []
    c = base_channels
    
    for i in range(num_conv_layers):
        # 1. Choose Block Type
        layer_type = 'inception' if use_inception else 'conv'
        
        # 2. Define Layer Config
        layer_cfg = {
            'type': layer_type,
            'out': c,
            'bn': use_bn,
            'dropout': dropout_prob if use_dropout else 0.0,
            'residual': use_residual
        }
        config.append(layer_cfg)
        
        # 3. Add Attention if requested (e.g., after every block or specifically placed)
        if use_attention:
            config.append({'type': 'attention'})

        # 4. Add Pooling every 2 layers to reduce spatial size
        if i % 2 != 0: 
            config.append({'type': 'maxpool'})
            c *= 2 # Double channels after pooling

    # Print the generated architecture for verification
    print(f"--- Generating Model with {len(config)} config steps ---")
    
    return ModularCNN(num_classes=num_classes, config=config)


# ==========================================
# 5. USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Simulate an experiment from your flowchart
    
    # "Initial test with 3 convolutional layers"
    model_exp1 = create_experiment_model(num_classes=10, num_conv_layers=3, use_bn=False)
    
    # "Add batch normalization + Residuals"
    model_exp2 = create_experiment_model(num_classes=10, num_conv_layers=4, use_bn=True, use_residual=True)
    
    # "Test inception blocks + Attention"
    model_exp3 = create_experiment_model(
        num_classes=10, 
        num_conv_layers=3, 
        base_channels=64,
        use_inception=True, 
        use_attention=True
    )

    # Test Forward Pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model_exp3(dummy_input)
    print(f"Output shape: {output.shape}")

    # Test GradCAM (Mock)
    # cam = model_exp3.extract_grad_cam(dummy_input)