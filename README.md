# Comparative Analysis of Deep Learning Models

## Project Overview

This project conducts a comparative analysis of various deep learning models, including Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs). It utilizes popular frameworks like PyTorch and Keras (with TensorFlow backend) to implement and evaluate these models on diverse datasets. The primary goal is to understand the strengths and weaknesses of different architectures and frameworks in handling specific types of tasks, such as regression and image classification. The project involves data loading and preprocessing, model definition, training, evaluation, and visualization of results.

## Project Structure

The project is organized into several Python scripts and supporting files:

*   **`data_loader.py`**: Handles the loading and preprocessing of datasets (FashionMNIST, CIFAR-100, California Housing).
*   **`keras_models.py`**: Defines the Keras CNN architecture.
*   **`main.py`**: Main execution script for training, evaluation, and comparative analysis.
*   **`pytorch_models.py`**: Defines PyTorch ANN architectures for regression and classification.
*   **`train_eval.py`**: Contains functions for model training and evaluation for both PyTorch and Keras models.
*   **`visualization.py`**: Includes functions for visualizing training progress (learning curves, loss curves).
*   **`Comparative-Analysis-of-Machine-Learning-Models.pdf`**: The project report/paper providing a detailed analysis and findings.
*   **`Member_Distribution.md`**: A document detailing the specific contributions of each team member to the project.

## How to Run

Follow these steps to set up and run the project:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Atif1299/ANN-CNN-Neural-Studio.git
    cd ANN-CNN-Neural-Studio
    ```

2.  **Install dependencies:**
    Ensure you have Python installed. Then, install the required libraries using pip:
    ```bash
    pip install torch torchvision tensorflow scikit-learn matplotlib numpy
    ```

3.  **Run the main script:**
    Execute the `main.py` script to start the training and evaluation process:
    ```bash
    python main.py
    ```

## Datasets

The project utilizes the following datasets for model training and evaluation:

*   **California Housing Dataset:** A regression dataset used to predict housing prices based on various features. The PyTorch `RegressionANN` is trained on this dataset.
*   **CIFAR-100 Dataset:** A comprehensive image classification dataset with 100 classes, each containing 600 images (500 training, 100 testing). The Keras CNN model and the PyTorch `ClassificationANN` are trained and evaluated on this dataset.
*   **FashionMNIST Dataset:** An image classification dataset consisting of 70,000 grayscale images (60,000 training, 10,000 testing) of 10 different clothing categories. While `data_loader.py` includes functionality to load FashionMNIST, the `main.py` script uses CIFAR-100 for the PyTorch `ClassificationANN`.

## Models and Training

The following deep learning models are implemented and analyzed:

1.  **PyTorch Regression ANN (`RegressionANN`)**
    *   **Architecture:** A feedforward neural network designed for regression tasks.
    *   **Dataset:** California Housing
    *   **Training Parameters:**
        *   Learning Rate: 0.01
        *   Epochs: 50
        *   Batch Size: 64

2.  **PyTorch Classification ANN (`ClassificationANN`)**
    *   **Architecture:** A feedforward neural network tailored for image classification.
    *   **Dataset:** CIFAR-100
    *   **Training Parameters:**
        *   Learning Rate: 0.01
        *   Epochs: 30
        *   Batch Size: 64

3.  **Keras Convolutional Neural Network (CNN)**
    *   **Architecture:** A CNN designed for image classification tasks, leveraging convolutional and pooling layers.
    *   **Dataset:** CIFAR-100
    *   **Training Parameters:**
        *   Learning Rate: 0.001 (using Adam optimizer)
        *   Epochs: 50
        *   Batch Size: 64

## Evaluation Metrics

Model performance is assessed using the following metrics:

*   **Regression Models (e.g., PyTorch `RegressionANN`):**
    *   Mean Squared Error (MSE)
    *   Mean Absolute Error (MAE)
    *   R-squared (R²)

*   **Classification Models (e.g., PyTorch `ClassificationANN`, Keras CNN):**
    *   Accuracy
    *   Precision (macro-averaged)
    *   Recall (macro-averaged)
    *   F1-Score (macro-averaged)
    *   Confusion Matrix

## Visualizations

The project generates several visualizations to aid in understanding model performance and training dynamics:

*   **Learning Curves:** Plots showing training and validation accuracy over epochs.
*   **Loss Curves:** Plots depicting training and validation loss over epochs.
*   **Confusion Matrix:** Visual representation of classification performance, showing correct and incorrect predictions for each class.

These visualizations are generated by functions in `visualization.py`.

## Contributions

**Member 1: Talha Asif_42**
*   Implemented the PyTorch ANN models (both `RegressionANN` and `ClassificationANN` in `pytorch_models.py`).
*   Developed the training and evaluation logic for the PyTorch models (`train_pytorch_model` and `evaluate_model` in `train_eval.py`).
*   Created the data loading and preprocessing pipeline for the FashionMNIST and regression datasets (`load_pytorch_dataset`, and `load_regression_dataset` in `data_loader.py`).
*   Conducted initial data analysis, and hyper parameter selection for Pytorch Models.

**Member 2: Muhammad Atif_31**
*   Developed the Keras CNN model architecture (`create_keras_cnn` in `keras_models.py`).
*   Implemented the Keras CNN training and evaluation functions (`train_keras_cnn` and `evaluate_keras_cnn` in `train_eval.py`).
*   Created the data loading and preprocessing pipeline for the CIFAR-100 dataset (`load_keras_dataset` and `load_classification_data` in `data_loader.py`).
*   Implemented the visualization tools (`plot_learning_curves` and `plot_loss_curves` in `visualization.py`).
*   Conducted initial data analysis and hyper parameter selection for the Keras Model.
*   Oversaw the final comparative analysis in `main.py`.

## Important Notes

*   **Data Paths:** You might need to adjust the data paths in `data_loader.py` if you store the datasets in a custom location. The scripts currently assume the datasets will be downloaded to a default directory (e.g., `./data` or as per the PyTorch/Keras default download behavior).
*   **PyTorch Model Training Time:** The training times reported for PyTorch models in the output comparison table might not be perfectly accurate due to the method of timing used.
*   **TensorFlow/Keras GPU:** Ensure your environment is correctly configured if you intend to use GPU acceleration with TensorFlow/Keras.

This README provides a comprehensive guide to understanding, running, and contributing to the project. For more detailed information on the analysis and results, please refer to the `Comparative-Analysis-of-Machine-Learning-Models.pdf` document.
