import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import glob


from typing import *

class BOVW():
    
    def __init__(self, detector_type="AKAZE", codebook_size:int=50, 
                 detector_kwargs:dict={}, codebook_kwargs:dict={}, 
                 pca_dim: int = None, pyramid_levels:int = 1,
                 normalize:bool = False, method: str = "None", scalar:bool = False):

        self.detector_type = detector_type
        self.detector_kwargs = detector_kwargs
        self.codebook_size = codebook_size
        self.pca_dim = pca_dim
        self.pca_model = None
        self.pyramid = pyramid_levels > 1
        self.pyramid_levels = pyramid_levels
        self.normalize = normalize
        self.scalar = scalar
        self.method = method
        self.scaling = method != "None"

        # Initialize Detectors
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(**detector_kwargs)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create(**detector_kwargs)
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(**detector_kwargs)
        elif detector_type == 'DENSE_SIFT':
            # Dense SIFT uses SIFT to describe points we create manually
            self.detector = cv2.SIFT_create()
        else:
            raise ValueError("Detector must be 'SIFT', 'AKAZE', 'ORB', or 'DENSE_SIFT'")
        
        self.codebook_algo = MiniBatchKMeans(n_clusters=self.codebook_size, **codebook_kwargs)
        
    def _extract_features(self, image: Literal["H", "W", "C"]) -> Tuple:
        """
        Extracts features. 
        [cite_start]If DENSE_SIFT, generates a grid of keypoints manually [cite: 643-644].
        """
        if self.detector_type == 'DENSE_SIFT':
            # Get parameters (defaults: step=15, scales=[4,8,12,16])
            step_size = self.detector_kwargs.get('step_size', 15)
            scales = self.detector_kwargs.get('scales', [4, 8, 12, 16])
            
            keypoints = []
            h, w = image.shape[:2]
            
            # Generate the Grid
            for scale in scales:
                for y in range(step_size, h - step_size, step_size):
                    for x in range(step_size, w - step_size, step_size):
                        # Use positional arguments for cv2.KeyPoint to prevent crashes
                        kp = cv2.KeyPoint(float(x), float(y), float(scale))
                        keypoints.append(kp)
            
            # Compute descriptors for forced points
            return self.detector.compute(image, keypoints)
        else:
            # Standard detection
            return self.detector.detectAndCompute(image, None)
    
    def _update_fit_codebook(self, descriptors: List[np.ndarray]):
        all_descriptors = np.vstack(descriptors)
        
        # [cite_start]Apply PCA if requested [cite: 757-759]
        if self.pca_dim is not None:
            if self.pca_model is None:
                print(f"Training PCA to reduce to {self.pca_dim} dimensions...")
                self.pca_model = PCA(n_components=self.pca_dim)
                self.pca_model.fit(all_descriptors)
            all_descriptors = self.pca_model.transform(all_descriptors)

        self.codebook_algo.partial_fit(all_descriptors)
        return self.codebook_algo, self.codebook_algo.cluster_centers_
    
    def _compute_codebook_descriptor(self, descriptors: np.ndarray, kmeans) -> np.ndarray:
        codebook_descriptor = np.zeros(kmeans.n_clusters)
        
        if descriptors is not None and descriptors.shape[0] > 0:
            # Transform test descriptors using trained PCA
            if self.pca_dim is not None and self.pca_model is not None:
                descriptors = self.pca_model.transform(descriptors)
                
            visual_words = kmeans.predict(descriptors)
            for label in visual_words:
                codebook_descriptor[label] += 1
            
            # L2 Normalize histogram
            norm = np.linalg.norm(codebook_descriptor)
            if norm > 0:
                codebook_descriptor /= norm
        
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

