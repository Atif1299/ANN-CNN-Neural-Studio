import torch
from torchvision import datasets, transforms
from tensorflow.keras.datasets import cifar100, fashion_mnist
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy as np


data_path = "C:\\Users\\talha\\AI lab tasks\\data\\cifar10\\train"
data_path1= "C:\\Users\\talha\\AI lab tasks\\data\\cifar10\\test"



def load_pytorch_dataset(dataset_name):
    if dataset_name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_path1, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def load_keras_dataset(dataset_name):
    if dataset_name == 'cifar100':
        (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    return (train_images, train_labels), (test_images, test_labels)

def load_regression_dataset():
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

def load_classification_data(batch_size=64, dataset='CIFAR100'):
    if dataset == 'CIFAR100':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root=data_path1, train=False, download=True, transform=transform)
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
