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
# CLASSIFICATION HEADS
# ==========================================

class GlobalAvgPoolHead(nn.Module):
    """Original: Global Average Pooling + Linear"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MaxPoolHead(nn.Module):
    """Global Max Pooling + Linear (zero additional parameters vs avg pool)"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MixedPoolHead(nn.Module):
    """Concatenates Global Average + Max Pooling, then Linear"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(in_channels * 2, num_classes)
    
    def forward(self, x):
        avg = self.avg_pool(x)
        max_p = self.max_pool(x)
        x = torch.cat([avg, max_p], dim=1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MLPHead(nn.Module):
    """Shallow MLP: GAP -> FC -> ReLU -> Dropout -> FC"""
    def __init__(self, in_channels, num_classes, hidden_dim=512, dropout_prob=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x


class GeMPoolHead(nn.Module):
    """Generalized Mean Pooling + Linear (1 learnable param per channel)"""
    def __init__(self, in_channels, num_classes, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        # GeM pooling: (1/HW * sum(x^p))^(1/p)
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.pow(1.0 / self.p)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class AttentionPoolHead(nn.Module):
    """Learnable attention weights across spatial dimensions + Linear"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        # Compute attention weights
        attn_weights = self.attention(x)  # [B, 1, H, W]
        
        # Apply attention and pool
        x = x * attn_weights
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ==========================================
# THE MODULAR MODEL WRAPPER
# ==========================================

class ModularCNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 3, 
                 config: List[Dict] = None, head_type: str = 'gap',
                 head_params: Dict = None):
        """
        Args:
            num_classes: Output dimension.
            input_channels: Channels in input image (usually 3).
            config: A list of dictionaries describing the layers.
            head_type: Type of classification head. Options:
                - 'gap': Global Average Pooling (original)
                - 'max': Global Max Pooling
                - 'mixed': Mixed Average + Max Pooling
                - 'mlp': Shallow MLP head
                - 'gem': Generalized Mean Pooling
                - 'attention': Attention Pooling
            head_params: Optional dictionary of parameters for the head (e.g., hidden_dim for MLP)
        """
        super().__init__()
        self.config = config
        self.head_type = head_type
        
        # 1. Build Feature Extractor
        self.features, last_channel_count = self._build_features(input_channels, config)
        
        # 2. Build Classifier Head
        head_params = head_params or {}
        self.classifier = self._build_classifier_head(
            head_type, last_channel_count, num_classes, head_params
        )

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
            
            elif l_type == 'maxpool':
                k = layer_cfg.get('k', 2)
                s = layer_cfg.get('s', 2)
                block = nn.MaxPool2d(kernel_size=k, stride=s)
                out_c = current_channels 

            elif l_type == 'attention':
                block = SEAttention(current_channels)
                out_c = current_channels

            # Wrap with Residual if requested
            if residual and l_type not in ['maxpool', 'attention']:
                block = ResidualWrapper(block, current_channels, out_c)

            if block:
                layers.append(block)
                current_channels = out_c

        return nn.Sequential(*layers), current_channels

    def _build_classifier_head(self, head_type, in_channels, num_classes, head_params):
        """Factory method to create different classification heads"""
        
        if head_type == 'gap':
            return GlobalAvgPoolHead(in_channels, num_classes)
        
        elif head_type == 'max':
            return MaxPoolHead(in_channels, num_classes)
        
        elif head_type == 'mixed':
            return MixedPoolHead(in_channels, num_classes)
        
        elif head_type == 'mlp':
            hidden_dim = head_params.get('hidden_dim', 256)
            dropout_prob = head_params.get('dropout_prob', 0.5)
            return MLPHead(in_channels, num_classes, hidden_dim, dropout_prob)
        
        elif head_type == 'gem':
            p = head_params.get('p', 3.0)
            return GeMPoolHead(in_channels, num_classes, p=p)
        
        elif head_type == 'attention':
            return AttentionPoolHead(in_channels, num_classes)
        
        else:
            raise ValueError(f"Unknown head_type: {head_type}. Choose from: "
                           "'gap', 'max', 'mixed', 'mlp', 'gem', 'attention'")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_grad_cam(self, input_image: torch.Tensor, 
                         target_layer_idx: int = -1, 
                         target_category: int = None):
        """Extracts GradCAM heatmap."""
        if GradCAMPlusPlus is None:
            raise ImportError("pytorch_grad_cam not installed.")

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
            if x.dim() == 4 and x.shape[2] > 1:
                maps.append(x)
                names.append(f"Layer_{idx}_{layer.__class__.__name__}")
        return maps, names


# ==========================================
# EXPERIMENT HELPER
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
    use_attention: bool = False,
    head_type: str = 'gap',
    head_params: Dict = None
) -> ModularCNN:
    """
    Factory function to generate a model based on high-level experiment flags.
    
    New Args:
        head_type: Classification head type ('gap', 'max', 'mixed', 'mlp', 'gem', 'attention')
        head_params: Optional parameters for the head (e.g., {'hidden_dim': 256} for MLP)
    """
    
    config = []
    c = base_channels
    
    for i in range(num_conv_layers):
        layer_type = 'inception' if use_inception else 'conv'
        
        layer_cfg = {
            'type': layer_type,
            'out': c,
            'bn': use_bn,
            'dropout': dropout_prob if use_dropout else 0.0,
            'residual': use_residual
        }
        config.append(layer_cfg)
        
        if use_attention:
            config.append({'type': 'attention'})

        if i % 2 != 0: 
            config.append({'type': 'maxpool'})
            c *= 2

    print(f"--- Generating Model with {len(config)} config steps and '{head_type}' head ---")
    
    return ModularCNN(num_classes=num_classes, config=config, 
                     head_type=head_type, head_params=head_params)


# ==========================================
# USAGE EXAMPLES
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Different Classification Heads")
    print("=" * 60)
    
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # 1. Original GAP Head
    print("\n1. Global Average Pooling Head")
    model_gap = create_experiment_model(num_classes=10, num_conv_layers=3, head_type='gap')
    out = model_gap(dummy_input)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_gap.parameters()):,}")
    
    # 2. Max Pool Head
    print("\n2. Global Max Pooling Head")
    model_max = create_experiment_model(num_classes=10, num_conv_layers=3, head_type='max')
    out = model_max(dummy_input)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_max.parameters()):,}")
    
    # 3. Mixed Pool Head
    print("\n3. Mixed (Avg+Max) Pooling Head")
    model_mixed = create_experiment_model(num_classes=10, num_conv_layers=3, head_type='mixed')
    out = model_mixed(dummy_input)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_mixed.parameters()):,}")
    
    # 4. MLP Head
    print("\n4. MLP Head (with custom hidden dim)")
    model_mlp = create_experiment_model(
        num_classes=10, 
        num_conv_layers=3, 
        head_type='mlp',
        head_params={'hidden_dim': 256, 'dropout_prob': 0.3}
    )
    out = model_mlp(dummy_input)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_mlp.parameters()):,}")
    
    # 5. GeM Pool Head
    print("\n5. GeM Pooling Head")
    model_gem = create_experiment_model(num_classes=10, num_conv_layers=3, head_type='gem')
    out = model_gem(dummy_input)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_gem.parameters()):,}")
    
    # 6. Attention Pool Head
    print("\n6. Attention Pooling Head")
    model_attn = create_experiment_model(num_classes=10, num_conv_layers=3, head_type='attention')
    out = model_attn(dummy_input)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_attn.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("All heads tested successfully!")
    print("=" * 60)