import time
from data_loader import load_pytorch_dataset, load_keras_dataset, load_regression_dataset, load_classification_data
from pytorch_models import RegressionANN, ClassificationANN
from keras_models import create_keras_cnn
from train_eval import train_pytorch_model, evaluate_pytorch_model, evaluate_keras_cnn, train_keras_cnn
from visualization import plot_learning_curves, plot_loss_curves
from tensorflow.keras.datasets import cifar100
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    print("Starting PyTorch ANN Regression Training...")
   
    reg_lr = 0.01
    reg_epochs = 50
    reg_batch_size = 64

    X_train, X_test, y_train, y_test = load_regression_dataset()
    reg_model = RegressionANN()
    optimizer = torch.optim.Adam(reg_model.parameters(), lr=reg_lr)
    criterion = torch.nn.MSELoss()
    train_pytorch_model(reg_model, torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=reg_batch_size, shuffle=True), criterion, optimizer, reg_epochs)
    
    reg_model.eval()
    with torch.no_grad():
        y_pred = reg_model(X_test)
    reg_mse = mean_squared_error(y_test.numpy(), y_pred.squeeze().numpy())
    reg_mae = mean_absolute_error(y_test.numpy(), y_pred.squeeze().numpy())
    reg_r2 = r2_score(y_test.numpy(), y_pred.squeeze().numpy())
    print(f"PyTorch ANN Regression Results - MSE: {reg_mse:.4f}, MAE: {reg_mae:.4f}, R²: {reg_r2:.4f}\n")

    print("Starting PyTorch ANN Classification Training on CIFAR-100...")
    class_input_size = 3 * 32 * 32
    class_output_size = 100  
    class_lr = 0.01
    class_epochs = 30
    class_batch_size = 64

    train_loader, val_loader_class = load_classification_data(batch_size=class_batch_size, dataset='CIFAR100')
    class_model = ClassificationANN(class_input_size, class_output_size)
    optimizer = torch.optim.Adam(class_model.parameters(), lr=class_lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_pytorch_model(class_model, train_loader, criterion, optimizer, epochs=class_epochs)
    loss, accuracy = evaluate_pytorch_model(class_model, val_loader_class)
    print(f"PyTorch ANN Classification Results - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%\n")

    print("Starting Keras CNN Classification Training on CIFAR-100...")
    (X_train_keras, y_train_keras), (X_val_keras, y_val_keras) = cifar100.load_data()
    X_train_keras = X_train_keras.astype('float32') / 255.0
    X_val_keras = X_val_keras.astype('float32') / 255.0

    keras_model = create_keras_cnn(input_shape=(32, 32, 3), num_classes=100)
    keras_model, keras_history, keras_time = train_keras_cnn(keras_model, X_train_keras, y_train_keras, X_val_keras, y_val_keras,
                                                              lr=0.001, epochs=50, batch_size=64)
    keras_accuracy, keras_cm, keras_precision, keras_recall, keras_f1 = evaluate_keras_cnn(keras_model, X_val_keras, y_val_keras)
    print(f"Keras CNN Classification Results - Accuracy: {keras_accuracy:.2f}%, Precision: {keras_precision:.4f}, Recall: {keras_recall:.4f}, F1-Score: {keras_f1:.4f}, Training Time: {keras_time:.2f} seconds\n")

    print("Comparative Table:")
    print(f"{'Model':<25}{'Dataset / Task':<25}{'Key Hyperparams':<50}{'Final Metric':<60}{'Training Time'}")
    print(f"{'PyTorch ANN (Regressor)':<25}{'California Housing':<25}{'LR=0.01, Epoch=50, Batch=64':<50}{f'MSE={reg_mse:.2f}; MAE={reg_mae:.2f}; R²={reg_r2:.2f}':<60}{f'~{time.time() - time.time():.2f} min'}")
    print(f"{'PyTorch ANN (Classifier)':<25}{'CIFAR-100':<25}{'LR=0.01, Epoch=30, Batch=64':<50}{f'Accuracy={accuracy:.2f}%':<60}{f'~{time.time() - time.time():.2f} min'}")
    print(f"{'Keras CNN (Classifier)':<25}{'CIFAR-100':<25}{'LR=0.001, Epoch=50, Batch=64':<50}{f'Accuracy={keras_accuracy:.2f}%':<60}{f'~{keras_time/60:.2f} min'}")

if __name__ == "__main__":
    main()