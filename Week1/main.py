from bovw import BOVW

from typing import *
from PIL import Image

import numpy as np
import glob
import tqdm
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import time


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"]):
    return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])

def pyramid_extract_bovw_histograms(bovw: Type[BOVW], all_descriptors: Literal["N", "T", "d"],all_keypoints,all_dimensions):
    all_top_left = []
    all_top_right = []
    all_bottom_left = []
    all_bottom_right = []
    
    for descriptors,keypoints,dimensions in zip(all_descriptors,all_keypoints,all_dimensions):
        desc_top_left = []
        desc_top_right = []
        desc_bottom_left = []
        desc_bottom_right = []
        
        h,w = dimensions
        
        for kp, desc in zip(keypoints, descriptors):
            x, y = kp.pt

            if y < h/2:  
                if x < w/2: 
                    desc_top_left.append(desc)
                else:       
                    desc_top_right.append(desc)
            else:     
                if x < w/2:  
                    desc_bottom_left.append(desc)
                else:        
                    desc_bottom_right.append(desc)
        all_top_left.append(np.array(desc_top_left))
        all_top_right.append(np.array(desc_top_right))
        all_bottom_left.append(np.array(desc_bottom_left))
        all_bottom_right.append(np.array(desc_bottom_right))
        

    global_histograms = extract_bovw_histograms(descriptors=all_descriptors, bovw=bovw) 
    top_left_histograms = extract_bovw_histograms(descriptors=all_top_left, bovw=bovw)
    top_right_histograms = extract_bovw_histograms(descriptors=all_top_right, bovw=bovw)
    bottom_left_histograms = extract_bovw_histograms(descriptors=all_bottom_left, bovw=bovw)
    bottom_right_histograms = extract_bovw_histograms(descriptors=all_bottom_right, bovw=bovw)
    
    return np.concatenate([global_histograms,top_left_histograms,top_right_histograms,bottom_left_histograms,bottom_right_histograms],axis=1)

    

def test(dataset: List[Tuple[Type[Image.Image], int]]
         , bovw: Type[BOVW], 
        
         classifier:Type[object], 
         pyramid: bool = True,):
    
    test_descriptors = []
    descriptors_labels = []
    
    test_keypoints = []
    test_dimensions = []
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Eval]: Extracting the descriptors"):
        image, label = dataset[idx]
        keyponts, descriptors = bovw._extract_features(image=np.array(image))
        
        height,width = image.height, image.width

            
        if descriptors is not None:
            test_descriptors.append(descriptors)
            descriptors_labels.append(label)
            
            test_dimensions.append((height,width))
            test_keypoints.append(keyponts)
            
    if pyramid:
        print("Computing the pyramid bovw histograms")
        
        bovw_histograms = pyramid_extract_bovw_histograms(all_descriptors=test_descriptors, bovw=bovw,all_keypoints=test_keypoints,all_dimensions=test_dimensions)
        
    
    else:
        print("Computing the bovw histograms")
        bovw_histograms = extract_bovw_histograms(descriptors=test_descriptors, bovw=bovw)
        
    print("predicting the values")
    y_pred = classifier.predict(bovw_histograms)
    
    acc = accuracy_score(y_true=descriptors_labels, y_pred=y_pred)
    print("Accuracy on Phase[Test]:", acc)
    return acc


def train(dataset: List[Tuple[Type[Image.Image], int]],
           bovw:Type[BOVW],
           pyramid:bool = True):
    all_descriptors = []
    all_keypoints = []
    all_labels = []
    img_dimensions = []
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Training]: Extracting the descriptors"):
        
        image, label = dataset[idx]
        height,width = image.height, image.width
        keyponts, descriptors = bovw._extract_features(image=np.array(image))
        
        if descriptors  is not None:
            all_descriptors.append(descriptors)
            all_labels.append(label)
            
            img_dimensions.append((height,width))
            all_keypoints.append(keyponts)
            
    print("Fitting the codebook")
    kmeans, cluster_centers = bovw._update_fit_codebook(descriptors=all_descriptors)

    

    if pyramid:
        print("Computing the pyramid bovw histograms")
        bovw_histograms = pyramid_extract_bovw_histograms(all_descriptors=all_descriptors, bovw=bovw,all_keypoints=all_keypoints,all_dimensions=img_dimensions)

        
    else:        

        print("Computing the bovw histograms")
        bovw_histograms = extract_bovw_histograms(descriptors=all_descriptors, bovw=bovw) 
    
    print("Fitting the classifier")
    classifier = LogisticRegression(class_weight="balanced",max_iter=10_000).fit(bovw_histograms, all_labels)

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
    
    start_time = time.time()
    
     #/home/cboned/data/Master/MIT_split
    data_train = Dataset(ImageFolder="../places_reduced/train")
    data_test = Dataset(ImageFolder="../places_reduced/val") 

    pyramid = True

    bovw = BOVW()
    
    bovw, classifier = train(dataset=data_train, bovw=bovw,pyramid=pyramid)
    
    test(dataset=data_test, bovw=bovw, classifier=classifier,pyramid=pyramid)
    
    end_time = time.time()
    print("Pyramid_", pyramid)
    print("Elapsed time:", end_time-start_time)
