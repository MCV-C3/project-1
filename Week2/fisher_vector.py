"""
Outputs:
CSV log: fisher_vector_results.csv
saved GMM models (joblib)
"""

import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed, dump, load
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC,LinearSVC


from bovw import BOVW
from old_utils import Dataset


os.environ['JOBLIB_TEMP_FOLDER'] = '/home/msiau/workspace/jventosa/PostTFG/Master/project-1/Week2/joblib'

# Utility functions
def gather_all_descriptors(dataset, bovw: BOVW, max_descriptors=200000, per_image_limit=None,pca_value = None):
    """
    Extract descriptors for all images in dataset using bovw._extract_features.
    Returns:
      descriptors_list: list of np.ndarray descriptors per image (only non-None)
      all_desc_sample: stacked descriptors sampled for GMM fitting (up to max_descriptors)
      labels: list of labels aligned with descriptors_list
    """
    descriptors_list = []
    labels = []
    sampled = []

    for img, label in tqdm(dataset, desc="Extracting descriptors"):
        _, desc = bovw._extract_features(image=np.array(img))
        if desc is None:
            continue
        # to speed up and preserve randomness
        if per_image_limit is not None and desc.shape[0] > per_image_limit:
            # uniform subsample
            idxs = np.random.choice(desc.shape[0], per_image_limit, replace=False)
            desc_use = desc[idxs]
        else:
            desc_use = desc

        descriptors_list.append(desc_use)
        labels.append(label)

        sampled.append(desc_use)

    if len(sampled) == 0:
        raise ValueError("No descriptors found in dataset. Check detector settings or dataset content.")

    # Stacking all descriptors and randomly sample up to max_descriptors for GMM training
    all_desc = np.vstack(sampled)
    if all_desc.shape[0] > max_descriptors:
        idxs = np.random.choice(all_desc.shape[0], max_descriptors, replace=False)
        all_desc_sample = all_desc[idxs]
    else:
        all_desc_sample = all_desc


    return descriptors_list, labels, all_desc_sample


def fit_gmm(descriptors, n_components=64, cov_type='diag', random_state=42, save_path=None):
    """
    Fitting a GaussianMixture using sklearn (diagonal covariances).
    descriptors: n x d array (sampled)
    Returns fitted gmm.
    """
    print("y")
    print(f"Fitting GMM with K={n_components}, cov_type={cov_type} on {descriptors.shape[0]} descriptors...")
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type=cov_type,
                          max_iter=300,
                          verbose=1,
                          reg_covar=1e-3,
                          random_state=random_state)
    gmm.fit(descriptors)
    if save_path:
        dump(gmm, save_path)
        print("Saved GMM to:", save_path)
    return gmm


def fisher_vector(descriptors, gmm: GaussianMixture, eps=1e-8):
    """
    Compute Fisher Vector for one image's descriptors.
    descriptors: M x D
    gmm: fitted GaussianMixture with attributes weights_, means_, covariances_
    Returns: 1D numpy array of shape (2 * K * D,)
    """
    # defensive checks
    if descriptors is None or descriptors.shape[0] == 0:
        # Return zero vector of appropriate dim
        K = gmm.n_components
        D = gmm.means_.shape[1]
        return np.zeros(2 * K * D, dtype=np.float32)

    # (N x K)
    q = gmm.predict_proba(descriptors)  # q_ik
    N = descriptors.shape[0]
    K = gmm.n_components
    D = gmm.means_.shape[1]

    # Get GMM params
    w = gmm.weights_ + eps  # shape (K,)
    mu = gmm.means_         # (K, D)
    # For diag covariances, covariances_ shape (K, D)
    sigma = np.sqrt(gmm.covariances_) if gmm.covariance_type == 'diag' else np.sqrt(np.array([np.diag(c) for c in gmm.covariances_]))

    # Precompute denominators
    # Normalization terms used
    # u_k = 1/(N * sqrt(w_k)) * sum_i q_ik * ((x_i - mu_k) / sigma_k)
    # v_k = 1/(N * sqrt(2 w_k)) * sum_i q_ik * ( ((x_i - mu_k)**2 / sigma_k**2) - 1 )
    sqrt_w = np.sqrt(w)
    sqrt_2w = np.sqrt(2.0 * w)

    # Compute sums per component
    # vectorize: for each k, compute sums over i
    # Expand shapes to allow broadcasting
    # descriptors: (N, D), mu: (K, D), sigma: (K, D), q: (N, K)

    # Transpose q for broadcasting: (K, N)
    q_T = q.T  # (K, N)

    # Compute first order (u) and second order (v)
    u = np.zeros((K, D), dtype=np.float64)
    v = np.zeros((K, D), dtype=np.float64)

    # vectorized approach for efficiency: for each k compute weighted sums
    for k in range(K):
        qk = q_T[k][:, np.newaxis]  # (N,1)
        diff = (descriptors - mu[k]) / (sigma[k] + eps)  # (N, D)
        u_k = (qk * diff).sum(axis=0)  # (D,)
        sq_diff = (diff ** 2)
        v_k = (qk * (sq_diff - 1.0)).sum(axis=0)  # (D,)

        u[k] = u_k
        v[k] = v_k

    # Normalize by N and sqrt(w_k) terms
    u = (u / N) / (sqrt_w[:, np.newaxis] + eps)
    v = (v / N) / (sqrt_2w[:, np.newaxis] + eps)

    # Concatenate [u_0, u_1, ..., u_{K-1}, v_0, v_1, ..., v_{K-1}] into 1D
    fv = np.concatenate([u.ravel(), v.ravel()]).astype(np.float32)

    # Power norm
    fv = np.sign(fv) * np.sqrt(np.abs(fv) + 1e-12)
    # L2 norm
    norm = np.linalg.norm(fv) + 1e-12
    fv = fv / norm

    return fv


def compute_fvs_for_dataset(descriptors_list, gmm):
    """
    descriptors_list: list of arrays (per image)
    Returns X (n_images x fv_dim)
    """
    X = Parallel(n_jobs=-1)(
        delayed(fisher_vector)(desc, gmm) 
        for desc in tqdm(descriptors_list, desc="Computing Fisher Vectors (Parallel)")
    )
    
    return np.vstack(X)



# Main experiment flow
def run_fv_experiments(train_folder="../places_reduced/train",
                       test_folder="../places_reduced/val",
                       detector_type="SIFT",
                       gaussian_list=[32, 64, 128],
                       max_gmm_samples=200000,
                       per_image_limit=None,
                       random_state=42,
                       save_models=True):
    """
    Runs FV experiments for different numbers of Gaussians and logs results.
    """
    print("Loading datasets...")
    train_dataset = Dataset(ImageFolder=train_folder)
    test_dataset = Dataset(ImageFolder=test_folder)

    print(f"Using detector: {detector_type}")
    # instantiate a BOVW object only for descriptor extraction without kmeans
    bovw_for_desc = BOVW(detector_type=detector_type, codebook_size=64,
                         detector_kwargs={'step_size': 15, 'scales':[8]} if detector_type=='DENSE_SIFT' else {})

    print("Gathering descriptors from training set (this may take a while)...")
    descriptors_list_train, labels_train, sample_desc = gather_all_descriptors(
        train_dataset, bovw_for_desc,
        max_descriptors=max_gmm_samples,
        per_image_limit=per_image_limit
    )
    print("Total train images with descriptors:", len(descriptors_list_train))
    print("Sample descriptors shape (for GMM):", sample_desc.shape)

    results_records = []

    # Fitting GMMs for each gaussian count and evaluate
    for K in gaussian_list:
        print("\n" + "="*60)
        print(f"Experiment: GMM components K = {K}")
        print("="*60)

        model_name = f"gmm_K{K}_det{detector_type}.joblib"
        # Fitting GMM or loading if exists
        if save_models and os.path.exists(model_name):
            print("Loading existing GMM:", model_name)
            gmm = load(model_name)
        else:
            gmm = fit_gmm(sample_desc, n_components=K, cov_type='diag', random_state=random_state,
                          save_path=model_name if save_models else None)

        # Compute Fisher Vectors for training set
        X_train = compute_fvs_for_dataset(descriptors_list_train, gmm)
        y_train = np.array(labels_train)

        # Also compute FVs for test set
        print("Extracting descriptors for test set...")
        descriptors_list_test = []
        labels_test = []
        for img, label in tqdm(test_dataset, desc="Extract descriptors test"):
            _, desc = bovw_for_desc._extract_features(image=np.array(img))
            if desc is None:
                continue
            if per_image_limit is not None and desc.shape[0] > per_image_limit:
                idxs = np.random.choice(desc.shape[0], per_image_limit, replace=False)
                desc = desc[idxs]
            descriptors_list_test.append(desc)
            labels_test.append(label)

        X_test = compute_fvs_for_dataset(descriptors_list_test, gmm)
        y_test = np.array(labels_test)


        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train classifier (Logistic Regression) and evaluate with CV and test
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
        print("Cross-validating on training FVs (5-fold)...")
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
        mean_cv = cv_scores.mean()
        std_cv = cv_scores.std()
        print(f"CV accuracy: {mean_cv:.4f} ± {std_cv:.4f}")

        # Fit on all training and evaluate on test set
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"Test accuracy on FV features (K={K}): {test_acc:.4f}")

        # Log results
        results_records.append({
            "timestamp": pd.Timestamp.now(),
            "detector": detector_type,
            "K": K,
            "n_train_images": len(X_train),
            "n_test_images": len(X_test),
            "cv_mean_acc": float(mean_cv),
            "cv_std_acc": float(std_cv),
            "test_acc": float(test_acc)
        })

        # save intermediate CSV
        df = pd.DataFrame(results_records)
        df.to_csv("fisher_vector_results.csv", index=False)
        print("Saved results to fisher_vector_results.csv")

    print("\nAll experiments complete. Final results:")
    print(pd.DataFrame(results_records))
    return results_records

def neural_based_fisher(train_descriptors, test_descriptors, train_labels, test_labels):



    sample_for_gmm = train_descriptors.reshape((-1,train_descriptors.shape[2]))
    idxs = np.random.choice(sample_for_gmm.shape[0], 2000, replace=False)
    sample_for_gmm = sample_for_gmm[idxs]

    gmm = fit_gmm(sample_for_gmm, n_components=128, cov_type='diag', random_state=42,
                   )


    X_train = compute_fvs_for_dataset(train_descriptors, gmm)
    y_train = np.array(train_labels)

    # Also compute FVs for test set
    print("Extracting descriptors for test set...")
    

    X_test = compute_fvs_for_dataset(test_descriptors, gmm)
    y_test = np.array(test_labels)


    scaler = Normalizer(norm='l2')
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("mamaMia")
    # Train classifier (Logistic Regression) and evaluate with CV and test
    clf = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=100, n_jobs=4)
    print("pizzeria")
    clf.fit(X_train_scaled, y_train)
    print("Fitted")
    y_pred = clf.predict(X_test_scaled)
    print("Predicted")
    test_acc = accuracy_score(y_test, y_pred)

    return test_acc, (y_pred,y_test)

def final_mod3(train_dataset,test_dataset):

    


    # instantiate a BOVW object only for descriptor extraction without kmeans
    bovw_for_desc = BOVW(detector_type="DENSE_SIFT", codebook_size=128, method="l2",detector_kwargs={'step_size': 15, 'scales': [4, 8, 12, 16]})

    print("Gathering descriptors from training set (this may take a while)...")
    descriptors_list_train, labels_train, sample_desc = gather_all_descriptors(
        train_dataset, bovw_for_desc,
        max_descriptors=200000,
        per_image_limit=500,
        pca_value=None
    )
    print("Total train images with descriptors:", len(descriptors_list_train))
    print("Sample descriptors shape (for GMM):", sample_desc.shape)



    # Fitting GMMs for each gaussian count and evaluate

    print(sample_desc.shape)
    gmm = fit_gmm(sample_desc, n_components=128, cov_type='diag', random_state=42,
                   )


    X_train = compute_fvs_for_dataset(descriptors_list_train, gmm)
    y_train = np.array(labels_train)

    # Also compute FVs for test set
    print("Extracting descriptors for test set...")
    descriptors_list_test = []
    labels_test = []
    for img, label in tqdm(test_dataset, desc="Extract descriptors test"):
        _, desc = bovw_for_desc._extract_features(image=np.array(img))
        if desc is None:
            continue
        
        descriptors_list_test.append(desc)
        labels_test.append(label)

    X_test = compute_fvs_for_dataset(descriptors_list_test, gmm)
    y_test = np.array(labels_test)


    scaler = Normalizer(norm='l2')
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(X_train_scaled.shape)
    
    # Train classifier (Logistic Regression) and evaluate with CV and test
    clf = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=15000, n_jobs=-1)
    print("Cross-validating on training FVs (5-fold)...")
    print(X_train_scaled.shape)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy',n_jobs=-1)
    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()
    print(f"CV accuracy: {mean_cv:.4f} ± {std_cv:.4f}")

    # Fit on all training and evaluate on test set
    clf.fit(X_train_scaled, y_train)
    print("Fitted")
    y_pred = clf.predict(X_test_scaled)
    print("Predicted")
    test_acc = accuracy_score(y_test, y_pred)

    return test_acc, (y_pred,y_test)


if __name__ == "__main__":
    records = run_fv_experiments(
        train_folder="../places_reduced/train",
        test_folder="../places_reduced/val",
        detector_type="DENSE_SIFT",
        gaussian_list=[32, 64, 128],
        max_gmm_samples=200000,
        per_image_limit=500,  # None for no limit
        random_state=42,
        save_models=True
    )

