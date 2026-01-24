import cv2
from bovw import BOVW

from typing import *
from PIL import Image

import numpy as np
import glob
import tqdm
import os
import pickle
import optuna
import pandas as pd
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.cluster import MiniBatchKMeans

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler



def Dataset(ImageFolder:str = "data/MIT_split/train") -> List[Tuple[Type[Image.Image], int]]:
    map_classes = {clsi: idx for idx, clsi  in enumerate(os.listdir(ImageFolder))}
    print(map_classes)
    dataset = []
    for idx, cls_folder in enumerate(os.listdir(ImageFolder)):
        
        image_path = os.path.join(ImageFolder, cls_folder)
        images = glob.glob(image_path+"/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")
            dataset.append((img_pil, map_classes[cls_folder]))
    return dataset