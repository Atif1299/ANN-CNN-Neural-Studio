import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from tensorflow.keras.optimizers import Adam
import time

def train_pytorch_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0

    with torch.no_grad():  
        for data, target in test_loader:
            output = model(data) 
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            correct += (output.argmax(1) == target).sum().item()

    average_loss = total_loss / len(test_loader.dataset)  
    accuracy = 100.0 * correct / len(test_loader.dataset) 
    return average_loss, accuracy

def train_keras_cnn(model, X_train, y_train, X_val, y_val, lr, epochs, batch_size):
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    start_time = time.time()
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop],
                        verbose=0)
    end_time = time.time()
    training_time = end_time - start_time
    return model, history, training_time

def evaluate_keras_cnn(model, X_val, y_val):
    predictions = model.predict(X_val)
    predicted_classes = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(y_val, predicted_classes) * 100
    cm = confusion_matrix(y_val, predicted_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, predicted_classes, average='weighted')
    
    return accuracy, cm, precision, recall, f1

def evaluate_pytorch_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy