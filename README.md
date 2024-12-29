# Comparative Analysis of Machine Learning Models

This project implements and compares different machine learning models for regression and classification tasks using PyTorch and Keras. The models include Artificial Neural Networks (ANNs) for both regression and classification using PyTorch, and a Convolutional Neural Network (CNN) for classification using Keras.

## Project Structure

The project is organized into the following Python files:

*   **`data_loader.py`**: This file handles the loading and preprocessing of datasets. It includes functions to load the following datasets:
    *   **FashionMNIST**: Loads the FashionMNIST dataset for PyTorch.
    *   **CIFAR100**: Loads the CIFAR-100 dataset for Keras and PyTorch.
    *   **California Housing**: Loads the California Housing dataset for regression.
*   **`main.py`**: This is the main execution file. It orchestrates the training and evaluation of all models. It loads data, creates model instances, defines optimizers, sets the loss function, trains models, evaluates models using the specified metrics and also provides comparative analysis of all the models.
*   **`keras_models.py`**: This file defines the Keras CNN architecture used for classification. The function `create_keras_cnn` constructs a CNN model.
*   **`pytorch_models.py`**: This file defines the PyTorch ANN architectures. It includes the following classes:
    *   `RegressionANN`: Defines an ANN model for regression.
    *   `ClassificationANN`: Defines an ANN model for classification.
*   **`train_eval.py`**: This file contains functions for training and evaluating the models.
    *   `train_pytorch_model`: Trains PyTorch models.
    *   `evaluate_model`: Evaluates PyTorch classification models.
    *   `train_keras_cnn`: Trains Keras CNN models.
    *   `evaluate_keras_cnn`: Evaluates Keras CNN models and calculates metrics.
*   **`visualization.py`**: This file includes functions for visualizing training progress.
    *   `plot_learning_curves`: Plots training and validation accuracy curves.
    *  `plot_loss_curves`: Plots training and validation loss curves.

## How to Run the Code

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Atif1299/ANN-CNN-Neural-Studio.git
    cd ANN-CNN-Neural-Studio
    ```
2.  **Install the required libraries:**

    ```bash
    pip install torch torchvision tensorflow scikit-learn matplotlib numpy
    ```
3.  **Run the main script:**

    ```bash
    python main.py
    ```

    The script will automatically download the necessary datasets and train/evaluate the models.

## Datasets

The project uses the following datasets:

*   **California Housing Dataset:** For regression tasks. It has 8 input features.
*   **CIFAR-100 Dataset:** For multi-class image classification (100 classes). Images are 32x32 RGB.
*   **FashionMNIST Dataset**: For multi-class image classification (10 classes). Images are 28x28 Grayscale.

## Models and Training

The project includes the following models:

*   **PyTorch Regression ANN:** A feed-forward network for predicting median house values in California Housing dataset. It is trained using the Mean Squared Error loss function.
*  **PyTorch Classification ANN:** A feed-forward network for classifying images in the CIFAR100 dataset. It is trained using the Cross Entropy loss function.
*   **Keras CNN:** A convolutional neural network for classifying images in the CIFAR-100 dataset. It is trained using the Sparse Categorical Cross Entropy loss function.

Training parameters for each model:

* **PyTorch Regression ANN:**
    * Learning Rate: 0.01
    * Epochs: 50
    * Batch Size: 64
* **PyTorch Classification ANN:**
   * Learning Rate: 0.01
    * Epochs: 30
    * Batch Size: 64
* **Keras CNN:**
    * Learning Rate: 0.001
    * Epochs: 50
    * Batch Size: 64

## Evaluation Metrics

The models are evaluated using the following metrics:

*   **Regression (PyTorch ANN):**
    *   Mean Squared Error (MSE)
    *   Mean Absolute Error (MAE)
    *   R-squared (R²)
*   **Classification (PyTorch ANN):**
    *   Loss
    *   Accuracy
*   **Classification (Keras CNN):**
    *   Accuracy
    *   Precision
    *   Recall
    *   F1-Score
    *  Confusion Matrix
    *  Training Time

## Visualizations

The project generates learning curves, which display the training and validation accuracy as well as loss of the Keras and Pytorch classifier. The project also displays the confusion matrix obtained from the Keras CNN.

## Notes

*   The data paths in `data_loader.py` are hardcoded. You may need to adjust these paths to match your local directory structure.
*  The training times of the Pytorch models displayed in the final table may not be accurate.







```python

```
