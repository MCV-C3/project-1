# README for Baseline Implementation

## Project Overview

This project implements a baseline model for Bag of Visual Words (BoVW) with various local descriptors. The model is used to extract visual features from images and classify them using a logistic regression model. It supports descriptors such as **SIFT**, **ORB**, and **AKAZE**.

## Requirements
- Python 3.8+
- Required Python libraries:
  - OpenCV (`cv2`)
  - Numpy (`numpy`)
  - Scikit-learn (`scikit-learn`)
  - PIL (`Pillow`)
  - Matplotlib (`matplotlib`)
  - tqdm (`tqdm`)

To install the required packages, run:
```bash
pip install numpy opencv-python scikit-learn pillow matplotlib tqdm
```

## Code Structure
1. **`bovw.py`**: Contains the `BOVW` class, responsible for feature extraction, codebook creation, and histogram computation.
2. **`main.py`**: Main script to load datasets, train the BoVW model, and evaluate its performance.

## Usage

### Dataset Preparation
The dataset should follow this structure:
```
data/
  ├── MIT_split/
  │   ├── train/
  │   │   ├── <class_label>/
  │   │   │   ├── img1.jpg
  │   │   │   ├── img2.jpg
  │   ├── test/
  │       ├── <class_label>/
  │       │   ├── img1.jpg
```
Replace `<class_label>` with appropriate labels.

### Run the Script
To execute the baseline model:
```bash
python main.py
```

### Training Process
- The `train` function fits a codebook and trains a logistic regression classifier on the BoVW histograms.
- The `test` function evaluates the classifier on the test set.

## Customization
- Modify `BOVW` initialization to change descriptors:
  ```python
  bovw = BOVW(detector_type="ORB", codebook_size=100)
  ```
- Add more parameters via `detector_kwargs` and `codebook_kwargs`.

## Results Table
Below is the baseline performance for different local descriptors:

| Local Descriptor Model | Accuracy Train | Accuracy Test |
|-------------------------|----------------|---------------|
| SIFT                   | 0.647          | 0.645         |
| ORB                    | 0.395          | 0.371         |
| AKAZE                  | 0.4689         | 0.4426        |

## Future Improvements
- Implement other machine learning models (e.g., SVM) for classification.
- Experiment with different codebook sizes.
- Add support for dense feature extraction.
```