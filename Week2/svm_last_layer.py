from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import SimpleModel
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm

from sklearn.svm import LinearSVC
from main import train,test,extract_features,train_with_patches,test_patches,extract_patch_features
from fisher_vector import neural_based_fisher
from IPython.display import clear_output
from torch.utils.data import TensorDataset, DataLoader

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="2"

os.environ['JOBLIB_TEMP_FOLDER'] = '/home/msiau/workspace/jventosa/PostTFG/Master/project-1/Week2/joblib'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

optimal_image_size = 16
transformation  = F.Compose([
                                F.ToImage(),
                                F.ToDtype(torch.float32, scale=True),
                                F.Resize(size=(optimal_image_size, optimal_image_size)),
                            ])

data_train = ImageFolder("../places_reduced/train", transform=transformation)
data_test = ImageFolder("../places_reduced/val", transform=transformation) 

train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8)
test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8)

model = SimpleModel(input_d=3*16*16,hidden_layers_n=1, hidden_d=256, output_d=11)

model = model.to(device)
model.load_state_dict(torch.load(f"SimpleModel/768_input_1_layers_256_dimension.pth", weights_only=True))

train_feats, train_labels = extract_features(model, train_loader, device, 1)
test_feats, test_labels = extract_features(model, test_loader, device, 1)

svm = LinearSVC(C=1.0)
svm.fit(train_feats, train_labels)
svm_acc = svm.score(test_feats, test_labels)

print(f"SVM accuracy using layer {1}: {svm_acc:.4f}")