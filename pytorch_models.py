import torch.nn as nn
import torch.nn.functional as F
import torch


class RegressionANN(nn.Module):
    def __init__(self):
        super(RegressionANN, self).__init__()
        self.weights1 = nn.Parameter(torch.randn(8, 16))  
        self.bias1 = nn.Parameter(torch.randn(16))

        self.weights2 = nn.Parameter(torch.randn(16, 8))  
        self.bias2 = nn.Parameter(torch.randn(8))

        self.weights3 = nn.Parameter(torch.randn(8, 4))  
        self.bias3 = nn.Parameter(torch.randn(4))

        self.weights4 = nn.Parameter(torch.randn(4, 1)) 
        self.bias4 = nn.Parameter(torch.randn(1))
        

    def forward(self, x):
        x = torch.matmul(x, self.weights1) + self.bias1
        x = torch.relu(x)
        x = torch.matmul(x, self.weights2) + self.bias2
        x = torch.relu(x)
        x = torch.matmul(x, self.weights3) + self.bias3
        x = torch.relu(x)
        x = torch.matmul(x, self.weights4) + self.bias4
        return x
        

class ClassificationANN(nn.Module):
    def __init__(self,input_size,ouput_size):
        super(ClassificationANN, self).__init__()
        self.weights1 = nn.Parameter(torch.randn(input_size, 512))  
        self.bias1 = nn.Parameter(torch.randn(512))  
        
        self.weights2 = nn.Parameter(torch.randn(512, 256))  
        self.bias2 = nn.Parameter(torch.randn(256))  
        
        self.weights3 = nn.Parameter(torch.randn(256, 128))  
        self.bias3 = nn.Parameter(torch.randn(128))  
        
        self.weights4 = nn.Parameter(torch.randn(128, ouput_size))  
        self.bias4 = nn.Parameter(torch.randn(ouput_size))  

    def forward(self, x):
         x = torch.flatten(x, start_dim=1) 
        
        
      
         x = torch.matmul(x, self.weights1) + self.bias1
         x = torch.relu(x)  
        
     
         x = torch.matmul(x, self.weights2) + self.bias2
         x = torch.relu(x) 
        
        
         x = torch.matmul(x, self.weights3) + self.bias3
         x = torch.relu(x)  
        
         x = torch.matmul(x, self.weights4) + self.bias4
        
        
         x = torch.softmax(x, dim=1) 
        
        
         return x 
           