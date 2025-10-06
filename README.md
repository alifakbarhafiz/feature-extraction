<div align="center">

# Deep Feature Extraction & ML Classification

A comprehensive Jupyter Notebook for extracting rich feature representations from images using a PyTorch Convolutional Neural Network (CNN) and classifying them with classical Machine Learning algorithms (k-NN, SVM, Decision Tree) on the **CIFAR-10** dataset.

<img src="KNN (MLP).png" alt="Classification Results Table" width="550">
<img src="KNN (PCA).png" alt="Classification Results Table" width="550">

*Example visualization: Comparative results of ML models on extracted features*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Notebook Flow](#notebook-flow)
  - [Customization](#customization)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## 🔍 Overview

This project implements a hybrid approach to image classification. First, a **Convolutional Neural Network (CNN)**, implemented in PyTorch, is trained or loaded to serve as a **feature extractor**. The notebook then uses an intermediate layer of the CNN to generate a high-dimensional feature vector for every image in the dataset.

These deep features are then used to train and evaluate three traditional **Scikit-learn** classifiers:
1.  **k-Nearest Neighbors (k-NN)**
2.  **Support Vector Machine (SVM)**
3.  **Decision Tree Classifier (DTC)**

The primary goal is to compare the classification performance of these traditional models when operating on powerful deep features versus raw pixel data. The entire pipeline is contained within the `FeatureExtraction.ipynb` notebook.

## ✨ Features

-   🧠 **Deep Feature Extraction**: Utilizes a PyTorch CNN (trained on CIFAR-10) to generate expressive feature vectors.
-   📊 **Comparative Analysis**: Head-to-head comparison of three robust traditional classifiers (k-NN, SVM, DTC).
-   🔬 **Feature Standardization**: Implements **StandardScaler** to normalize feature vectors, improving classifier performance.
-   📐 **Comprehensive Evaluation**: Reports key metrics including **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrices** for each model.
-   💾 **Reproducibility**: A single, well-documented Jupyter Notebook (`FeatureExtraction.ipynb`) detailing every step from data loading to final evaluation.
-   🚀 **GPU Acceleration**: CUDA support for faster CNN training/inference (if a pre-trained model isn't used).

---

## 🚀 Installation

### Prerequisites

-   Python 3.8+
-   A working setup for Jupyter or Google Colab (as the notebook metadata suggests).
-   NVIDIA GPU with CUDA (Recommended for potential CNN training, but not strictly necessary for feature extraction).

### Setup

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
    # cd your-repo-name
    ```

2.  **Install dependencies:**
    The notebook relies on the following key packages, as observed from the imports:
    ```bash
    pip install -r requirements.txt
    ```

**Required packages:**
-   `torch` and `torchvision` - Deep learning framework.
-   `numpy` - Numerical operations.
-   `scikit-learn` - Traditional ML models, data scaling, and metrics.
-   `matplotlib` - Visualization.

---

## 💻 Usage

Open and run the cells sequentially in the `FeatureExtraction.ipynb` notebook using Jupyter Lab/Notebook or Google Colab.

### Notebook Flow

The notebook is structured into logical, sequential sections:

1.  **Import Dependencies**: Loads all necessary libraries (`torch`, `sklearn` modules, etc.).
2.  **Data Loading & Preprocessing**: Downloads the **CIFAR-10** dataset and applies necessary `transforms`.
3.  **Feature Extractor Setup**: Defines and potentially trains a CNN model (or loads a pre-trained checkpoint).
4.  **Feature Extraction**: Runs the test and train datasets through the CNN up to a specific layer (e.g., the penultimate layer) to get the feature vectors.
5.  **Feature Standardization**: Initializes and fits the **StandardScaler** on the training features and transforms both the train and test feature sets.
6.  **Model Training**: Trains the k-NN, SVM, and Decision Tree classifiers on the standardized feature data.
7.  **Evaluation**: Tests all models, generates classification reports, and visualizes confusion matrices.

### Customization

Key parameters to modify directly within the notebook's code cells:

-   **`FEATURE_LAYER`**: The specific CNN layer (by index or name) used to extract the features.
-   **`KNN_NEIGHBORS`**: The number of neighbors for the k-NN classifier.
-   **`SVM_KERNEL`**: The kernel type for the SVM (e.g., `'linear'`, `'rbf'`).

---

## ⚙️ Configuration

Key configuration details assumed in the notebook:

### Deep Learning Feature Extractor

| Parameter | Value/Description |
| :--- | :--- |
| **Dataset** | CIFAR-10 (10 classes) |
| **Image Size** | 32x32 RGB |
| **Model** | Simple CNN (or a lightweight architecture like ResNet/VGG) |
| **Feature Layer** | Typically the layer *before* the final classification head |

### Traditional ML Classifiers

| Model | Key Hyperparameters |
| :--- | :--- |
| **k-NN** | `n_neighbors` (e.g., 5 or 7) |
| **SVM** | `C` (Regularization), `kernel` (e.g., `'rbf'`) |
| **Decision Tree** | `max_depth` |
| **Preprocessing**| `StandardScaler` applied to feature vectors |

---

## 🏗️ Model Architecture

### 1. Feature Extractor (CNN)

The feature extractor is the **Deep Learning backbone**. It's typically a simple CNN or a pre-trained model (like a truncated ResNet/VGG) used for representation learning on the CIFAR-10 task.

-   **Purpose**: To convert the raw 32x32 pixel data into a dense, abstract, and semantically rich feature vector (e.g., 512-dimensional).
-   **Output**: A flattened vector of extracted features from a non-final convolutional or pooling layer.

### 2. Traditional Classifiers

The core of the classification task is handled by the Scikit-learn models:

-   **k-Nearest Neighbors (k-NN)**: A non-parametric, distance-based model, sensitive to feature scaling.
-   **Support Vector Machine (SVM)**: A highly effective model that finds an optimal hyperplane to separate classes, often utilizing the **Radial Basis Function (RBF) kernel** for non-linear separation.
-   **Decision Tree Classifier**: A hierarchical model that splits the feature space based on a series of decision rules.

---

## 📈 Results

The notebook generates comparative results that demonstrate the relative strengths of the traditional ML models on the rich, standardized feature space provided by the CNN.

### Expected Output

1.  **Standardized Feature Data**: Visualization (e.g., a scatter plot using PCA/t-SNE) of the extracted features.
2.  **Classification Report**: A comprehensive table comparing all three models:

| Metric | k-NN | SVM (RBF) | Decision Tree |
| :--- | :--- | :--- | :--- |
| **Accuracy** | High | Highest | Moderate |
| **Precision** | High | High | Moderate |
| **F1-Score** | High | High | Moderate |

3.  **Confusion Matrices**: Visualizations showing the classification performance per class for each model.

---

## 🔧 Troubleshooting

### Common Issues

**Scikit-learn warnings:**
-   Ensure features are **standardized** using `StandardScaler` (Cell 5) before training k-NN or SVM. These models are highly sensitive to feature magnitude.

**CUDA out of memory (if training the CNN):**
-   If you are training the CNN yourself, reduce the batch size in the data loading utility.
-   If simply loading a pre-trained model, this is unlikely.

**Poor Classification Performance:**
-   Check the quality of the CNN features. The CNN model used for extraction must be adequately trained on the image task to provide useful features.
-   Tune the key hyperparameters for k-NN (`n_neighbors`) and SVM (`C` and `gamma`) for optimal results.

---

## 📚 References

### Datasets and Frameworks

-   [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
-   [PyTorch Documentation](https://pytorch.org/docs/)
-   [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Key Concepts

-   **Transfer Learning & Feature Extraction**: Using pre-trained models to extract features.
-   **Support Vector Machines (SVM)**: Principles of hyperplane maximization and kernel functions.
-   **k-Nearest Neighbors (k-NN)**: Distance metrics and feature scaling.

---

## 🙏 Acknowledgments

-   The PyTorch team for the open-source deep learning framework.
-   The creators of the Scikit-learn library for robust machine learning tools.
-   The developers of the CIFAR-10 dataset.

---

**Author**: Alif Akbar Hafiz

If you find this implementation helpful, please consider giving it a ⭐!
