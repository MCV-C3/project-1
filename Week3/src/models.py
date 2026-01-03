import torch.nn as nn
import torchvision.models as models

def build_model(name, num_classes, dropout):

    if name == "resnet18":
        model = models.resnet18(pretrained=True)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, num_classes)
        )

    elif name == "squeezenet":
        model = models.squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, 1)

    else:
        raise ValueError("Unknown model type")

    return model
