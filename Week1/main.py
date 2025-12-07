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

from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler

def apply_scaling(bovw_histograms, method="l2"):
    if method == "standard":
        scaler = StandardScaler()
        return scaler.fit_transform(bovw_histograms)
    elif method == "minmax":
        scaler = MinMaxScaler()
        return scaler.fit_transform(bovw_histograms)
    elif method == "l1":
        scaler = Normalizer(norm='l1')
        return scaler.transform(bovw_histograms)
    elif method == "l2":
        scaler = Normalizer(norm='l2')
        return scaler.transform(bovw_histograms)
    return bovw_histograms


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"]):
    return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])

def pyramid_extract_bovw_histograms(bovw: Type[BOVW], all_descriptors, all_keypoints, all_dimensions, levels=2):
    # [cite_start]Implements Spatial Pyramid Matching [cite: 667]
    pyramid_histograms = []
    
    for level in range(1, levels + 1):
        num_cells = level * level
        level_cells = [[[] for _ in range(len(all_descriptors))] for _ in range(num_cells)]
        
        for img_idx, (descriptors, keypoints, dimensions) in enumerate(zip(all_descriptors, all_keypoints, all_dimensions)):
            if descriptors is None or len(descriptors) == 0:
                continue

            h, w = dimensions
            w_step = w / level
            h_step = h / level
            
            for kp, desc in zip(keypoints, descriptors):
                x, y = kp.pt
                col = int(min(x // w_step, level - 1))
                row = int(min(y // h_step, level - 1))
                cell_index = row * level + col
                level_cells[cell_index][img_idx].append(desc)

        for cell_descriptors_per_image in level_cells:
            formatted_descs = [np.array(d) if len(d)>0 else np.zeros((0, 128)) for d in cell_descriptors_per_image]
            cell_hist = extract_bovw_histograms(descriptors=formatted_descs, bovw=bovw)
            pyramid_histograms.append(cell_hist)

    return np.concatenate(pyramid_histograms, axis=1)



def get_descriptors(dataset, bovw, cache_dir="cache"):
    """
    Smart Cache Logic:
    - DENSE_SIFT: Always recalculates (params vary).
    - SIFT/ORB/AKAZE: Loads from disk cache if exists.
    """
    # 1. Handling Dense SIFT
    if bovw.detector_type == 'DENSE_SIFT':
        print(f"Detector is DENSE_SIFT. Skipping cache to ensure correct parameters...")
        all_descriptors, all_keypoints, img_dimensions, all_labels = [], [], [], []
        
        for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Extracting Dense Features]"):
            image, label = dataset[idx]
            keypoints, descriptors = bovw._extract_features(image=np.array(image))
            
            if descriptors is not None:
                all_descriptors.append(descriptors)
                all_labels.append(label)
                img_dimensions.append((image.height, image.width))
                all_keypoints.append(keypoints)
                
        return all_descriptors, all_keypoints, img_dimensions, all_labels

    # 2. Handling Standard Detectors (Use Cache)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"train_descriptors_{bovw.detector_type}.pkl")
    
    if os.path.exists(cache_file):
        print(f"Loading cached descriptors from {cache_file}...")
        try:
            with open(cache_file, "rb") as f:
                all_descriptors, keypoints_data, img_dimensions, all_labels = pickle.load(f)
            
            # Reconstruct cv2.KeyPoints
            all_keypoints = []
            for img_kps in keypoints_data:
                kps = tuple([cv2.KeyPoint(x=p[0], y=p[1], size=p[2], angle=p[3], response=p[5], octave=p[4], class_id=p[6]) for p in img_kps])
                all_keypoints.append(kps)
                
            print(f"Loaded {len(all_descriptors)} images from cache.")
            return all_descriptors, all_keypoints, img_dimensions, all_labels
        except Exception as e:
            print(f"Cache load failed ({e}). Re-extracting...")

    print(f"No cache found for {bovw.detector_type}. Extracting features...")
    all_descriptors, all_keypoints, img_dimensions, all_labels = [], [], [], []
    
    for idx in tqdm.tqdm(range(len(dataset)), desc=f"Phase [Extracting {bovw.detector_type}]"):
        image, label = dataset[idx]
        keypoints, descriptors = bovw._extract_features(image=np.array(image))
        
        if descriptors is not None:
            all_descriptors.append(descriptors)
            all_labels.append(label)
            img_dimensions.append((image.height, image.width))
            all_keypoints.append(keypoints)

    # Save to cache (Convert keypoints to tuples)
    keypoints_to_pickle = []
    for img_kps in all_keypoints:
        tupled_kps = tuple([(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.octave, kp.response, kp.class_id) for kp in img_kps])
        keypoints_to_pickle.append(tupled_kps)

    with open(cache_file, "wb") as f:
        pickle.dump((all_descriptors, keypoints_to_pickle, img_dimensions, all_labels), f)
        
    return all_descriptors, all_keypoints, img_dimensions, all_labels

def manual_grid_search_codebook_size(all_descriptors, all_labels, detector_type="AKAZE", k_values=[50, 100]):
    print(f"\n--- Starting Manual Grid Search for K ---")
    results = []
    
    for k in k_values:
        print(f"Testing k={k}...")
        trial_bovw = BOVW(detector_type=detector_type, codebook_size=k)
        trial_bovw._update_fit_codebook(descriptors=all_descriptors)
        X_histograms = extract_bovw_histograms(trial_bovw, all_descriptors)
        
        # Fast Linear SVM for K search
        clf = LinearSVC(class_weight="balanced", dual="auto", max_iter=2000)
        scores = cross_val_score(clf, X_histograms, all_labels, cv=3, scoring='accuracy', n_jobs=None)
        
        results.append({"k": k, "mean_acc": scores.mean()})

    best_result = max(results, key=lambda x: x['mean_acc'])
    print(f"Best K found: {best_result['k']} (Acc: {best_result['mean_acc']:.4f})")
    return best_result['k']


def test(dataset: List[Tuple[Type[Image.Image], int]], 
         bovw: Type[BOVW], 
         classifier: Type[object]):
    
    test_descriptors = []
    descriptors_labels = []
    test_keypoints = []
    test_dimensions = []

    # 1. Extract Features from Test Set
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Eval]"):
        image, label = dataset[idx]
        
        # We need keypoints for Spatial Pyramid, descriptors for everything
        keypoints, descriptors = bovw._extract_features(image=np.array(image))
        
        if descriptors is not None:
            test_descriptors.append(descriptors)
            descriptors_labels.append(label)
            # Store dimensions/keypoints for Pyramid logic
            test_dimensions.append((image.height, image.width))
            test_keypoints.append(keypoints)
            
    # 2. Compute Histograms (Pyramid vs Standard)
    if bovw.pyramid:
        # print("Computing Pyramid Histograms (Test)...")
        bovw_histograms = pyramid_extract_bovw_histograms(
            bovw, 
            test_descriptors, 
            test_keypoints, 
            test_dimensions, 
            levels=bovw.pyramid_levels
        )
    else:
        # print("Computing Standard Histograms (Test)...")
        bovw_histograms = extract_bovw_histograms(bovw, test_descriptors)
        
    # 3. Apply Scaling (if enabled in BoVW object)
    if bovw.scaling:
        bovw_histograms = apply_scaling(bovw_histograms, bovw.method)
        
    # 4. Predict & Evaluate
    y_pred = classifier.predict(bovw_histograms)
    acc = accuracy_score(y_true=descriptors_labels, y_pred=y_pred)
    
    print(f"Test Accuracy: {acc:.4f}")
    
    # CRITICAL: Return the score so run_final_pipeline can save it!
    return acc

def evaluate_multiple_classifiers(X, y, cv=5, detector_type=None, codebook_size=None):

    # 1. Classifiers that NEED tuning (Linear models)
    # We will search for the best 'C' parameter for these
    tunable_classifiers = {
        "log_reg": LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000, n_jobs=None),
        "svm_linear": LinearSVC(class_weight="balanced", dual="auto", max_iter=2000) 
    }
    
    # 2. Classifiers that are usually fine with defaults (or too slow to tune in a loop)
    fixed_classifiers = {
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=None),
        "rf": RandomForestClassifier(n_estimators=100, n_jobs=None),
        # RBF SVM is slow, so we use fixed params or a very small cache
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced", cache_size=1000) 
    }

    results = {}
    
    # --- A. Tuning Loop (Recovers your lost accuracy) ---
    # Small grid to keep it fast but effective
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]} 
    
    for name, clf in tunable_classifiers.items():
        print(f"\nTuning & Evaluating: {name}...")
        
        # GridSearchCV automatically finds the best C using 3-fold CV
        gs = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=None)
        
        # nested cross_val_score evaluates the "process of finding the best C"
        scores = cross_val_score(gs, X, y, cv=cv, scoring="accuracy", n_jobs=None)
        
        mean_acc = scores.mean()
        std_acc = scores.std()
        print(f"  CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        results[name] = (mean_acc, std_acc)

    # --- B. Fixed Loop ---
    for name, clf in fixed_classifiers.items():
        print(f"\nEvaluating: {name}...")
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=None)
        
        mean_acc = scores.mean()
        std_acc = scores.std()
        print(f"  CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        results[name] = (mean_acc, std_acc)

    # Pick the best
    best_name = max(results, key=lambda k: results[k][0])
    best_mean, best_std = results[best_name]

    print("-" * 30)
    print(f"Best classifier: {best_name} with {best_mean:.4f}")
    print("-" * 30)

    # --- Logging ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Ensure values are not None for CSV
    d_type = detector_type if detector_type else "Unknown"
    c_size = codebook_size if codebook_size else 0
    
    df = pd.DataFrame([
        {
            "timestamp": timestamp,
            "detector_type": d_type,
            "codebook_size": c_size,
            "classifier": name,
            "mean_acc": acc[0],
            "std_acc": acc[1]
        }
        for name, acc in results.items()
    ])

    df.to_csv(
        "classifier_experiments_log.csv",
        mode="a",                                
        index=False,
        header=not os.path.exists("classifier_experiments_log.csv")  
    )

    return best_name, results

# --- MAIN TRAIN FUNCTION ---
def train(dataset, bovw: Type[BOVW], use_optimize: bool = True):
    
    # 1. Feature Extraction
    all_descriptors, all_keypoints, img_dimensions, all_labels = get_descriptors(dataset, bovw)
    
    det_type = "AKAZE"
    if "SIFT" in str(bovw.detector): det_type = "SIFT"
    elif "ORB" in str(bovw.detector): det_type = "ORB"

    # 2. Optimization
    if use_optimize:
        k_grid = [64, 128, 256, 512]
        best_k = manual_grid_search_codebook_size(all_descriptors, all_labels, detector_type=det_type, k_values=k_grid)
        bovw.codebook_size = best_k
        bovw.codebook_algo = MiniBatchKMeans(n_clusters=best_k, batch_size=2048, random_state=42)

    # 3. Final Codebook
    print(f"Fitting final codebook (k={bovw.codebook_size})...")
    bovw._update_fit_codebook(descriptors=all_descriptors)

    # 4. Histograms
    if bovw.pyramid:
        print(f"Computing Pyramid Histograms (Levels={bovw.pyramid_levels})...")
        bovw_histograms = pyramid_extract_bovw_histograms(bovw, all_descriptors, all_keypoints, img_dimensions, levels=bovw.pyramid_levels)
    else:
        print("Computing Standard Histograms...")
        bovw_histograms = extract_bovw_histograms(bovw, all_descriptors)

    # 5. Scaling
    if bovw.scaling:
        bovw_histograms = apply_scaling(bovw_histograms, bovw.method)

    # 6. Evaluation
    best_clf_name, clf_results = evaluate_multiple_classifiers(bovw_histograms, all_labels, cv=5)
    best_cv_score = clf_results[best_clf_name][0]
    
    # 7. Final Training
    print(f"Training final model ({best_clf_name})...")
    if best_clf_name == "log_reg": final_clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000)
    elif best_clf_name == "svm_linear": final_clf = LinearSVC(class_weight="balanced", dual="auto", max_iter=2000)
    elif best_clf_name == "svm_rbf": final_clf = SVC(kernel="rbf", class_weight="balanced", probability=True)
    elif best_clf_name == "knn": final_clf = KNeighborsClassifier(n_neighbors=5)
    elif best_clf_name == "rf": final_clf = RandomForestClassifier(n_estimators=100)
    
    final_clf.fit(bovw_histograms, all_labels)
    
    return bovw, final_clf, best_cv_score

def Dataset(ImageFolder:str = "data/MIT_split/train") -> List[Tuple[Type[Image.Image], int]]:
    map_classes = {clsi: idx for idx, clsi in enumerate(os.listdir(ImageFolder))}
    dataset = []
    for idx, cls_folder in enumerate(os.listdir(ImageFolder)):
        image_path = os.path.join(ImageFolder, cls_folder)
        images = glob.glob(image_path+"/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")
            dataset.append((img_pil, map_classes[cls_folder]))
    return dataset

# --- FINAL PIPELINE RUNNER ---
def run_final_pipeline(data_train, data_test):
    print("\n" + "="*50)
    print("STARTING FINAL TESTING PIPELINE")
    print("Strategy: Change ONE parameter at a time (Baseline: SIFT, k=128, LogReg)")
    print("="*50 + "\n")
    
    results = []
    
    # --- HELPER TO RUN AND LOG ---
    def run_experiment(exp_name, param_name, param_value, bovw_obj):
        print(f"\n[Experiment: {exp_name}] Testing {param_name} = {param_value}...")
        try:
            # 1. TRAIN on Training Data (Get CV Score)
            # This ensures the model is fitted on the training set
            trained_bovw, trained_clf, cv_score = train(data_train, bovw_obj, use_optimize=False)
            
            # 2. TEST on Test Data (Get Test Score)
            # This is critical for your report plots
            test_score = test(data_test, trained_bovw, trained_clf)
            
            # 3. Log BOTH scores
            results.append({
                "Experiment": exp_name, 
                "Parameter": param_name, 
                "Value": str(param_value), 
                "CV_Accuracy": cv_score,    # Validation Performance
                "Test_Accuracy": test_score # Generalization Performance
            })
            
            # Save incrementally
            pd.DataFrame(results).to_csv("final_experiment_results.csv", index=False)
            
        except Exception as e:
            print(f"!!! Error in {exp_name}: {e}")

    # 1. DETECTOR COMPARISON
    for det in ["SIFT", "ORB", "AKAZE"]:
        bovw = BOVW(detector_type=det, codebook_size=128)
        run_experiment("Detectors", "Detector", det, bovw)

    # 2. CODEBOOK SIZE
    for k in [64, 128, 256, 512, 1024]:
        bovw = BOVW(detector_type="SIFT", codebook_size=k)
        run_experiment("Codebook Size", "k", k, bovw)

    # 3. DENSE SIFT STEPS
    for step in [30, 20, 15]:
        bovw = BOVW(detector_type="DENSE_SIFT", codebook_size=128, detector_kwargs={'step_size': step, 'scales': [8]})
        run_experiment("Dense SIFT Step", "Step Size", step, bovw)

    # 4. DENSE SIFT SCALES
    scale_configs = [([4], "Small"), ([16], "Large"), ([4, 8, 12, 16], "Multi")]
    for scales, name in scale_configs:
        bovw = BOVW(detector_type="DENSE_SIFT", codebook_size=128, detector_kwargs={'step_size': 15, 'scales': scales})
        run_experiment("Dense SIFT Scales", "Scales", name, bovw)

    # 5. PCA (Dimensionality Reduction)
    for dim in [None, 80, 64, 32]:
        val = dim if dim else "128 (Original)"
        bovw = BOVW(detector_type="SIFT", codebook_size=128, pca_dim=dim)
        run_experiment("PCA", "Dimensions", val, bovw)

    # 6. SPATIAL PYRAMIDS
    for lvl in [1, 2, 3]:
        bovw = BOVW(detector_type="SIFT", codebook_size=128, pyramid_levels=lvl)
        run_experiment("Spatial Pyramid", "Levels", lvl, bovw)

    # 7. SCALING / NORMALIZATION (Added this)
    for method in ["None", "l1", "l2", "standard", "minmax", "hellinger"]:
        bovw = BOVW(detector_type="SIFT", codebook_size=128, method=method)
        run_experiment("Scaling", "Method", method, bovw)

    print("\n" + "="*50)
    print("PIPELINE COMPLETE. Final results saved to final_experiment_results.csv")
    print("="*50)
        
    

if __name__ == "__main__":
# 1. Load Train Data
    print("Loading Train Dataset...")
    data_train = Dataset(ImageFolder="../places_reduced/train")
    
    # 2. Load Test Data (This was missing)
    print("Loading Test Dataset...")
    data_test = Dataset(ImageFolder="../places_reduced/val") # or 'val' depending on your folder name
    
    # 3. Run the pipeline with BOTH datasets
    run_final_pipeline(data_train, data_test)


    
   


    
    
    