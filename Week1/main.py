from bovw import BOVW

from typing import *
from PIL import Image

import numpy as np
import glob
import tqdm
import os
import optuna
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.cluster import MiniBatchKMeans

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"]):
    return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])


def optimize_codebook_size(all_descriptors, all_labels, detector_type="AKAZE", n_trials=10):
    # Run optuna to find best k using cross-validation
    
    def objective(trial):
        
        #select the k value
        k = trial.suggest_int("k", 20, 200, step=20)
        
        #perform kmeans and fit the codebook
        trial_bovw = BOVW(detector_type=detector_type, codebook_size=k)
        trial_bovw._update_fit_codebook(descriptors=all_descriptors)
        
        # create the histograms
        x_histograms = extract_bovw_histograms(trial_bovw, all_descriptors)
        
        #cross-validate classifier
        clf = LogisticRegression(class_weight="balanced", solver="lbfgs")
        scores = cross_val_score(clf, x_histograms, all_labels, cv=5, scoring='accuracy')
        return scores.mean()
    
    #surpress optuna big logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    #log everything to keep the results
    results_df = study.trials_dataframe()
    
    # Save to CSV (e.g., "tuning_k_AKAZE.csv")
    csv_filename = f"experiment_log_k_{detector_type}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"Logged all {n_trials} trials to {csv_filename}")
    
    print(f"Best K found: {study.best_params['k']} with accuracy: {study.best_value:.4f}")
    print("-------------------------------------------------------")
    
    return study.best_params['k']

def evaluate_multiple_classifiers(X, y, cv=5, detector_type=None, codebook_size=None):

    classifiers = {
        "log_reg": LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000),
        "svm_linear": SVC(kernel="linear", class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced"),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "rf": RandomForestClassifier(n_estimators=100)
    }

    results = {}

    for name, clf in classifiers.items():

        print(f"\nEvaluating classifier: {name}")
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc = scores.std()
        print(f"  CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        results[name] = (mean_acc, std_acc)

    best_name = max(results, key=lambda k: results[k][0])  # highest mean acc
    best_mean, best_std = results[best_name]

    print("-" * 30)
    print(f"Best classifier: {best_name} with {best_mean:.4f} ± {best_std:.4f}")
    print("-" * 30)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([
        {
            "timestamp": timestamp,
            "detector_type": detector_type,
            "codebook_size": codebook_size,
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

    print("Saved classifier results to classifier_experiments_log.csv")

    return best_name, results


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
    
    print("Accuracy on Phase[Test]:", accuracy_score(y_true=descriptors_labels, y_pred=y_pred))
    

def train(dataset: List[Tuple[Type[Image.Image], int]],
          bovw: Type[BOVW],
          use_optimize: bool = True):

    all_descriptors = []
    all_labels = []

    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Training]: Extracting the descriptors"):

        image, label = dataset[idx]
        _, descriptors = bovw._extract_features(image=np.array(image))

        if descriptors is not None:
            all_descriptors.append(descriptors)
            all_labels.append(label)

    # --- Determine detector type string for logging/Optuna ---
    det_type = "AKAZE"  # default
    if hasattr(bovw.detector, "getDefaultName"):
        name = bovw.detector.getDefaultName()
        if "SIFT" in name:
            det_type = "SIFT"
        elif "ORB" in name:
            det_type = "ORB"
        else:
            det_type = "AKAZE"
    else:
        # fallback to checking string repr
        if "SIFT" in str(bovw.detector):
            det_type = "SIFT"
        elif "ORB" in str(bovw.detector):
            det_type = "ORB"
        else:
            det_type = "AKAZE"

    # --- OPTIONAL: OPTIMIZE CODEBOOK SIZE (k) WITH OPTUNA ---
    if use_optimize:
        best_k = optimize_codebook_size(
            all_descriptors,
            all_labels,
            detector_type=det_type,
            n_trials=10
        )

        # Update the main BOVW object with best k
        bovw.codebook_size = best_k
        bovw.codebook_algo = MiniBatchKMeans(
            n_clusters=best_k, batch_size=2048, random_state=42
        )

    # --- FINAL TRAINING (with current k, optimized or not) ---
    print(f"Fitting the final codebook (k={bovw.codebook_size})...")
    bovw._update_fit_codebook(descriptors=all_descriptors)

    print("Computing final histograms...")
    bovw_histograms = extract_bovw_histograms(descriptors=all_descriptors, bovw=bovw)

    # --- Evaluate multiple classifiers ---
    best_clf_name, clf_results = evaluate_multiple_classifiers(
        bovw_histograms,
        all_labels,
        cv=5,
        detector_type=det_type,
        codebook_size=bovw.codebook_size
    )
    print(f"Training final classifier: {best_clf_name}")

    # Get best CV score for logging / dense experiments
    best_cv_score = clf_results[best_clf_name][0]

    # --- Instantiate and train the final classifier according to best_clf_name ---
    if best_clf_name == "log_reg":
        final_clf = LogisticRegression(class_weight="balanced",
                                       solver="lbfgs",
                                       max_iter=2000)

    elif best_clf_name == "svm_linear":
        final_clf = SVC(kernel="linear",
                        class_weight="balanced",
                        probability=True)

    elif best_clf_name == "svm_rbf":
        final_clf = SVC(kernel="rbf",
                        class_weight="balanced",
                        probability=True)

    elif best_clf_name == "knn":
        final_clf = KNeighborsClassifier(n_neighbors=5)

    elif best_clf_name == "rf":
        final_clf = RandomForestClassifier(n_estimators=200)

    else:
        raise ValueError("Unknown classifier name returned")

    final_clf.fit(bovw_histograms, all_labels)

    # IMPORTANT: keep the same return signature so run_dense_experiments still works
    return bovw, final_clf, best_cv_score



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


def run_dense_experiments(dataset_train, dataset_test):
    results_log = []

    print("=== EXPERIMENT 1: Standard vs Dense SIFT ===")
    
    # 1. Standard SIFT (Baseline)
    print("\nRunning Standard SIFT...")
    bovw_std = BOVW(detector_type='SIFT', codebook_size=128)
    # We rely on the CV score returned by train()
    _, _, cv_score_std = train(dataset_train, bovw_std, use_optimize=False)
    
    results_log.append({
        "Experiment": "Type Comparison",
        "Detector": "Standard SIFT", 
        "Step Size": "N/A", 
        "Scales": "Default", 
        "CV Accuracy": cv_score_std
    })

    # 2. Dense SIFT - Step Size Analysis
    print("\nRunning Dense SIFT Step Sizes...")
    steps = [30, 20, 10] # Smaller step = more dense = usually better but slower
    
    for step in steps:
        print(f"Testing Step Size: {step}")
        bovw_dense = BOVW(
            detector_type='DENSE_SIFT', 
            codebook_size= 128, # Keep k fixed for fair comparison
            detector_kwargs={'step_size': step, 'scales': [8]}
        )
        _, _, cv_score = train(dataset_train, bovw_dense, use_optimize=False)
        
        results_log.append({
            "Experiment": "Step Size",
            "Detector": "Dense SIFT", 
            "Step Size": step, 
            "Scales": "[8]", 
            "CV Accuracy": cv_score
        })

    # 3. Dense SIFT - Scale Analysis
    print("\nRunning Dense SIFT Scale Analysis...")
    # Does scale play a role? We test small, large, and multi-scale.
    scale_configs = [ 
        ([4], "Small"), 
        ([16], "Large"), 
        ([4, 8, 12, 16], "Multi-Scale") 
    ]
    
    for scales, name in scale_configs:
        print(f"Testing Scales: {name} {scales}")
        bovw_dense = BOVW(
            detector_type='DENSE_SIFT', 
            codebook_size=128,
            detector_kwargs={'step_size': 15, 'scales': scales} # Fix step to 15
        )
        _, _, cv_score = train(dataset_train, bovw_dense, use_optimize=False)
        
        results_log.append({
            "Experiment": "Scale Analysis",
            "Detector": "Dense SIFT", 
            "Step Size": 15, 
            "Scales": str(scales), 
            "CV Accuracy": cv_score
        })

    # Save Results
    df = pd.DataFrame(results_log)
    df.to_csv("results_dense_sift.csv", index=False)
    print("\nExperiments Complete! Results saved to results_dense_sift.csv")
    print(df)
    


if __name__ == "__main__":
     #/home/cboned/data/Master/MIT_split
    print("Loading datasets...")
    data_train = Dataset(ImageFolder="../places_reduced/train")
    print("Train dataset loaded with", len(data_train), "images.")
    data_test = Dataset(ImageFolder="../places_reduced/val")
    print("Test dataset loaded with", len(data_test), "images.")

    # print("Results for SIFT detector:")
    # bovw = BOVW(detector_type='SIFT')
    # bovw, classifier = train(dataset=data_train, bovw=bovw, use_optimize=True)
    # test(dataset=data_test, bovw=bovw, classifier=classifier)
    
    run_dense_experiments(dataset_train=data_train, dataset_test=data_test)
    
    
    