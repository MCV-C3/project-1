import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import os
import glob


from typing import *

class BOVW():
    
    def __init__(self, detector_type="AKAZE", codebook_size:int=50, detector_kwargs:dict={}, codebook_kwargs:dict={}):

        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(**detector_kwargs)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create(**detector_kwargs)
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(**detector_kwargs)
        else:
            raise ValueError("Detector type must be 'SIFT', 'SURF', or 'ORB'")
        
        self.codebook_size = codebook_size
        self.codebook_algo = MiniBatchKMeans(n_clusters=self.codebook_size, **codebook_kwargs)
        
               
    ## Modify this function in order to be able to create a dense sift
    def _extract_features(self, image: Literal["H", "W", "C"]) -> Tuple:

        return self.detector.detectAndCompute(image, None)
    
    
    def _update_fit_codebook(self, descriptors: Literal["N", "T", "d"])-> Tuple[Type[MiniBatchKMeans],
                                                                               Literal["codebook_size", "d"]]:
        
        all_descriptors = np.vstack(descriptors)

        self.codebook_algo = self.codebook_algo.partial_fit(X=all_descriptors)

        return self.codebook_algo, self.codebook_algo.cluster_centers_
    
    def _compute_codebook_descriptor(self, descriptors: Literal["1 T d"], kmeans: Type[KMeans]) -> np.ndarray:

        visual_words = kmeans.predict(descriptors)
        
        
        # Create a histogram of visual words
        codebook_descriptor = np.zeros(kmeans.n_clusters)
        for label in visual_words:
            codebook_descriptor[label] += 1
        
        # Normalize the histogram (optional)
        codebook_descriptor = codebook_descriptor / np.linalg.norm(codebook_descriptor)
        
        return codebook_descriptor       
    




def visualize_bow_histogram(histogram, image_index, output_folder="./test_example.jpg"):
    """
    Visualizes the Bag of Visual Words histogram for a specific image and saves the plot to the output folder.
    
    Args:
        histogram (np.array): BoVW histogram.
        cluster_centers (np.array): Cluster centers (visual words).
        image_index (int): Index of the image for reference.
        output_folder (str): Folder where the plot will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(histogram)), histogram)
    plt.title(f"BoVW Histogram for Image {image_index}")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.xticks(range(len(histogram)))
    
    # Save the plot to the output folder
    plot_path = os.path.join(output_folder, f"bovw_histogram_image_{image_index}.png")
    plt.savefig(plot_path)
    
    # Optionally, close the plot to free up memory
    plt.close()

    print(f"Plot saved to: {plot_path}")

