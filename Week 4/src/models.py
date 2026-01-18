import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, use_bn=False, use_dropout=False, use_gap=False, depth=2):
        super().__init__()
        layers = []
        in_ch = 3
        ch = 32

        for _ in range(depth):
            layers.append(nn.Conv2d(in_ch, ch, 3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_ch = ch
            ch *= 2

        self.features = nn.Sequential(*layers)

        if use_gap:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_ch, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_ch * (128 // (2**depth))**2, 128),
                nn.ReLU(),
                nn.Dropout(0.5 if use_dropout else 0.0),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
