# Comparative Analysis of Deep Learning Models

## Project Overview

This project compares machine learning models for regression and classification using PyTorch and Keras. It includes PyTorch ANNs for both regression and classification, and a Keras CNN for image classification.

## Project Structure

The codebase is organized as follows:

*   **`data_loader.py`**: Handles dataset loading and preprocessing.
*   **`main.py`**: Main execution script for training and evaluation.
*   **`keras_models.py`**: Defines the Keras CNN architecture.
*   **`pytorch_models.py`**: Defines PyTorch ANN architectures for regression and classification.
*   **`train_eval.py`**: Contains functions for model training and evaluation.
*  **`visualization.py`**: Contains functions for generating visual graphs
*   **`README.md`**: Project documentation.

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Atif1299/ANN-CNN-Neural-Studio.git
    cd ANN-CNN-Neural-Studio
    ```

2.  **Install Dependencies:**
    ```bash
    pip install torch torchvision tensorflow scikit-learn matplotlib numpy
    ```

3.  **Execute the Main Script:**
    ```bash
    python main.py
    ```

## Key Project Details

*   **Datasets**: California Housing, CIFAR-100, FashionMNIST.
*   **Models**:
    *   PyTorch ANN (Regression): Trained using MSE.
    *   PyTorch ANN (Classification): Trained using Cross-Entropy.
    *   Keras CNN (Classification): Trained using Sparse Categorical Cross-Entropy.
* **Evaluation Metrics:** MSE, MAE, R-squared for Regression ; Accuracy, Loss for Pytorch Classification and Accuracy, Precision, Recall, F1-Score and confusion matrix for Keras CNN classification model.
*  **Visualizations:** Learning curves and Confusion Matrix generated using the script.

## Important Notes

*   Ensure all required libraries are installed (`requirements.txt`).
*   Data paths in `data_loader.py` may need adjustment if data is not downloaded correctly.
*   Pytorch model training time may not be accurate.

This README provides the necessary information to run the project, understand the codebase structure and key project implementation.
