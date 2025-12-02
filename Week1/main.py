from bovw import BOVW

from typing import *
from PIL import Image

import numpy as np
import glob
import tqdm
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"]):
    return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])


def test(dataset: List[Tuple[Type[Image.Image], int]]
         , bovw: Type[BOVW], 
         classifier:Type[object]):
    
    test_descriptors = []
    descriptors_labels = []
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Eval]: Extracting the descriptors"):
        image, label = dataset[idx]
        _, descriptors = bovw._extract_features(image=np.array(image))
        
        if descriptors is not None:
            test_descriptors.append(descriptors)
            descriptors_labels.append(label)
            
    
    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=test_descriptors, bovw=bovw)
    
    print("predicting the values")
    y_pred = classifier.predict(bovw_histograms)
    
    acc = accuracy_score(y_true=descriptors_labels, y_pred=y_pred)
    print("Accuracy on Phase[Test]:", acc)
    return acc


def train(dataset: List[Tuple[Type[Image.Image], int]],
           bovw:Type[BOVW]):
    all_descriptors = []
    all_labels = []
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Training]: Extracting the descriptors"):
        
        image, label = dataset[idx]
        _, descriptors = bovw._extract_features(image=np.array(image))
        
        if descriptors  is not None:
            all_descriptors.append(descriptors)
            all_labels.append(label)
            
    print("Fitting the codebook")
    kmeans, cluster_centers = bovw._update_fit_codebook(descriptors=all_descriptors)

    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=all_descriptors, bovw=bovw) 
    
    print("Fitting the classifier")
    classifier = LogisticRegression(class_weight="balanced").fit(bovw_histograms, all_labels)

    print("Accuracy on Phase[Train]:", accuracy_score(y_true=all_labels, y_pred=classifier.predict(bovw_histograms)))
    
    return bovw, classifier


def Dataset(ImageFolder:str = "data/MIT_split/train") -> List[Tuple[Type[Image.Image], int]]:

    """
    Expected Structure:

        ImageFolder/<cls label>/xxx1.png
        ImageFolder/<cls label>/xxx2.png
        ImageFolder/<cls label>/xxx3.png
        ...

        Example:
            ImageFolder/cat/123.png
            ImageFolder/cat/nsdf3.png
            ImageFolder/cat/[...]/asd932_.png
    
    """

    map_classes = {clsi: idx for idx, clsi  in enumerate(os.listdir(ImageFolder))}
    
    dataset :List[Tuple] = []

    for idx, cls_folder in enumerate(os.listdir(ImageFolder)):

        image_path = os.path.join(ImageFolder, cls_folder)
        images: List[str] = glob.glob(image_path+"/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")

            dataset.append((img_pil, map_classes[cls_folder]))


    return dataset


    


if __name__ == "__main__":
     #/home/cboned/data/Master/MIT_split
    data_train = Dataset(ImageFolder="../places_reduced/train")
    data_test = Dataset(ImageFolder="../places_reduced//test") 

    bovw = BOVW()
    
    bovw, classifier = train(dataset=data_train, bovw=bovw)
    
    test(dataset=data_test, bovw=bovw, classifier=classifier)