
import torch.nn as nn
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision import models
import matplotlib.pyplot as plt

from typing import *
from torchview import draw_graph
from graphviz import Source

from torchvision.models.squeezenet import Fire


from PIL import Image
import torchvision.transforms.v2  as F
import numpy as np 

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPoolingHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.attention_mask = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )
        
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        attn_weights = self.attention_mask(x) 

        b, c, h, w = attn_weights.shape
        attn_weights = F.softmax(attn_weights.view(b, c, -1), dim=-1)
        attn_weights = attn_weights.view(b, c, h, w)
        

        weighted_features = x * attn_weights
        pooled_features = weighted_features.sum(dim=(2, 3)) 
        
        return self.classifier(pooled_features)




class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class SimpleModel(nn.Module):

    def __init__(self, input_d: int, hidden_d: int, output_d: int):

        super(SimpleModel, self).__init__()

        self.input_d = input_d
        self.hidden_d = hidden_d
        self.output_d = output_d


        self.layer1 = nn.Linear(input_d, hidden_d)
        self.layer2 = nn.Linear(hidden_d, hidden_d)
        self.output_layer = nn.Linear(hidden_d, output_d)

        self.activation = nn.ReLU()


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)

        x = self.output_layer(x)
        
        return x
    


class WraperModel(nn.Module):
    def __init__(self, num_classes: int, feature_extraction: bool=True,batch_norm: bool=True, dropout: bool = True,dropout_prob: float = 0.5,classifier_type: str = "FCN"):
        super(WraperModel, self).__init__()

        self.num_classes = num_classes
        self.classifier_type = classifier_type


        # Load pretrained VGG16 model
        self.backbone = models.squeezenet1_0(weights='IMAGENET1K_V1')
        
        if batch_norm:
            self._add_batch_norm_to_backbone()


        if dropout:
            self.backbone.classifier[0] = nn.Dropout(p=dropout_prob)
        else:
            self.backbone.classifier[0] = nn.Identity()

        if feature_extraction:
            self.set_parameter_requires_grad(feature_extracting=feature_extraction)
        
                
                
        final_conv = self.backbone.classifier[1] 
        self.add_classifier(classifier_type,final_conv.in_channels, num_classes)


    def add_classifier(self,classifier_type,input_size, num_classes: int):
        if classifier_type == "FCN":
            self.backbone.classifier[1] = nn.Conv2d(
                in_channels=input_size,
                out_channels=num_classes,
                kernel_size=1
            )
        elif classifier_type == "MLP":
            self.backbone.classifier[1] = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif classifier_type == "Attention":
            self.backbone.classifier[1] = AttentionPoolingHead(
                in_channels=input_size, 
                num_classes=num_classes
            )

    def forward(self, x):
        return self.backbone(x)

    def add_squeeze_and_excite(self,reduction):
        new_layers = []

        for layer in self.backbone.features:
            new_layers.append(layer)

            if isinstance(layer, Fire):
                out_channels = (
                    layer.expand1x1.out_channels +
                    layer.expand3x3.out_channels
                )
                new_layers.append(SEBlock(out_channels, reduction))

        self.backbone.features = nn.Sequential(*new_layers)
        


    def add_fire_modules(self, n, sq_channels, exp_channels):

        
        for _ in range(n):
            # Get the input channels from the last module in the sequence
            in_channels = self._get_last_output_channels()
            
            new_fire = Fire(in_channels, sq_channels, exp_channels, exp_channels)
            
            self.backbone.features.add_module(f"fire_{len(self.backbone.features)}", new_fire)
        

    def delete_last_n_modules(self, n):

        new_features = list(self.backbone.features)[:-n]
        self.features = nn.Sequential(*new_features)
        self.add_classifier(self.classifier_type,self._get_last_output_channels(), self.num_classes)

    def _get_last_output_channels(self):

        last_layer = self.backbone.features[-1]
        print(last_layer)
        if isinstance(last_layer, Fire): # Specifically for Fire modules
            return last_layer.expand1x1.out_channels + \
                   last_layer.expand3x3.out_channels
        elif hasattr(last_layer, 'out_channels'):
            return last_layer.out_channels


    def _add_batch_norm_to_backbone(self):
        """
        Iterates through Fire modules and adds BN after the squeeze and expand convs.
        """
        for name, module in self.backbone.features.named_children():

            if isinstance(module, models.squeezenet.Fire):

                module.squeeze = nn.Sequential(
                    module.squeeze,
                    nn.BatchNorm2d(module.squeeze.out_channels)
                )

                module.expand1x1 = nn.Sequential(
                    module.expand1x1,
                    nn.BatchNorm2d(module.expand1x1.out_channels)
                )

                module.expand3x3 = nn.Sequential(
                    module.expand3x3,
                    nn.BatchNorm2d(module.expand3x3.out_channels)
                )
    

    def extract_feature_maps(self, input_image:torch.Tensor):

        conv_weights =[]
        conv_layers = []
        total_conv_layers = 0

        for module in self.backbone.features.children():
            if isinstance(module, nn.Conv2d):
                total_conv_layers += 1
                conv_weights.append(module.weight)
                conv_layers.append(module)


        print("TOTAL CONV LAYERS: ", total_conv_layers)
        feature_maps = []  # List to store feature maps
        layer_names = []  # List to store layer names
        x= torch.clone(input=input_image)
        for layer in conv_layers:
            x = layer(x)
            feature_maps.append(x)
            layer_names.append(str(layer))

        return feature_maps, layer_names



        

    def extract_features_from_hooks(self, x, layers: List[str]):
        """
        Extract feature maps from specified layers.
        Args:
            x (torch.Tensor): Input tensor.
            layers (List[str]): List of layer names to extract features from.
        Returns:
            Dict[str, torch.Tensor]: Feature maps from the specified layers.
        """
        outputs = {}
        hooks = []

        def get_activation(name):
            def hook(model, input, output):
                outputs[name] = output
            return hook

        # Register hooks for specified layers
        #for layer_name in layers:
        dict_named_children = {}
        for name, layer in self.backbone.named_children():
            for n, specific_layer in layer.named_children():
                dict_named_children[f"{name}.{n}"] = specific_layer

        for layer_name in layers:
            layer = dict_named_children[layer_name]
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))

        # Perform forward pass
        _ = self.forward(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return outputs

    def modify_layers(self, modify_fn: Callable[[nn.Module], nn.Module]):
        """
        Modify layers of the model using a provided function.
        Args:
            modify_fn (Callable[[nn.Module], nn.Module]): Function to modify a layer.
        """
        self.vgg16 = modify_fn(self.vgg16)


    def set_parameter_requires_grad(self, feature_extracting):
        """
        Set parameters gradients to false in order not to optimize them in the training process.
        """
        if feature_extracting:
            for param in self.backbone.parameters():
                param.requires_grad = False

        
        
    def remove_fire_blocks(self,n):
        fire_indices = [
            i for i, m in enumerate(self.backbone.features)
            if isinstance(m, Fire)
        ]

        remove_idxs = set(fire_indices[-n:])

        new_features = nn.Sequential(
            *[
                m for i, m in enumerate(self.backbone.features)
                if i not in remove_idxs
            ]
        )

        self.backbone.features = new_features
        
        self.add_classifier(self.classifier_type,self._get_last_output_channels(), self.num_classes)


    def extract_grad_cam(self, input_image: torch.Tensor, 
                         target_layer: List[Type[nn.Module]], 
                         targets: List[Type[ClassifierOutputTarget]]) -> Type[GradCAMPlusPlus]:

        

        with GradCAMPlusPlus(model=self.backbone, target_layers=target_layer) as cam:

            grayscale_cam = cam(input_tensor=input_image, targets=targets)[0, :]

        return grayscale_cam





# Example of usage
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load a pretrained model and modify it
    model = WraperModel(num_classes=8, feature_extraction=False)
    model.load_state_dict(torch.load("saved_model.pt"))
    #model = model

    """
        features.0
        features.2
        features.5
        features.7
        features.10
        features.12
        features.14
        features.17
        features.19
        features.21
        features.24
        features.26
        features.28
    """

    transformation  = F.Compose([
                                    F.ToImage(),
                                    F.ToDtype(torch.float32, scale=True),
                                    F.RandomHorizontalFlip(p=1.),
                                    F.Resize(size=(256, 256)),
                                ])
    # Example GradCAM usage
    dummy_input = Image.open("/home/msiau/data/tmp/jventosa/2425/MIT_large_train/test/highway/art803.jpg")#torch.randn(1, 3, 224, 224)
    input_image = transformation(dummy_input).unsqueeze(0)

    print(len(model.backbone.features))

    target_layers = [model.backbone.features[12]]
    targets = [ClassifierOutputTarget(6)]
    
    image = torch.from_numpy(np.array(dummy_input)).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min()) ## Image needs to be between 0 and 1 and be a numpy array (Remember that if you have norlized the image you need to denormalize it before applying this (image * std + mean))

    ## VIsualize the activation map from Grad Cam
    ## To visualize this, it is mandatory to have gradients.
    
    grad_cams = model.extract_grad_cam(input_image=input_image, target_layer=target_layers, targets=targets)

    visualization = show_cam_on_image(image, grad_cams, use_rgb=True)

    # Plot the result
    plt.imshow(visualization)
    plt.axis("off")
    plt.show()

    # Display processed feature maps shapes
    feature_maps, layer_names = model.extract_feature_maps(input_image)

                                                                 ### Aggregate the feature maps
    # Process and visualize feature maps
    processed_feature_maps = []  # List to store processed feature maps
    for feature_map in feature_maps:
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        min_feature_map, min_index = torch.min(feature_map, 0) # Get the min across channels
        processed_feature_maps.append(min_feature_map.data.cpu().numpy())
    
    
    # Plot All the convolution feature maps separately
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed_feature_maps)):
        ax = fig.add_subplot(5, 4, i + 1)
        ax.imshow(processed_feature_maps[i], cmap="hot", interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"{layer_names[i].split('(')[0]}_{i}", fontsize=10)

    plt.savefig("feature_maps.png")
    plt.show()

    ## Plot a concret layer feature map when processing a image thorugh the model
    ## Is not necessary to have gradients

    with torch.no_grad():
        feature_map = (model.extract_features_from_hooks(x=input_image, layers=["features.12"]))["features.12"]
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        print(feature_map.shape)
        processed_feature_map, _ = torch.min(feature_map, 0) 

    # Plot the result
    plt.imshow(processed_feature_map, cmap="gray")
    plt.axis("off")
    plt.savefig("processed_feature_map.png")
    plt.show()



    ## Draw the model
    # model_graph = draw_graph(model, input_size=(1, 3, 224, 224), device='meta', expand_nested=True, roll=True)
    # model_graph.visual_graph.render(filename="test", format="png", directory="./Week3")