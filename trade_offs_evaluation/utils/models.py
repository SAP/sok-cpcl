"""
Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0
"""

"""
Neural Network Models for DP Training Experiments

Provides plaintext and encrypted variants of neural networks for
differentially private training on MNIST/Fashion-MNIST datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import crypten.nn as cnn
from crypten.config import cfg

class ThreeLayerNN(nn.Module):
    """
    Three-layer feedforward neural network (plaintext PyTorch version).
    
    Architecture:
        Input (784) -> FC1 (hidden_size) -> ReLU
                   -> FC2 (hidden_size) -> ReLU
                   -> FC3 (output_size) -> Softmax
    
    Designed for MNIST/Fashion-MNIST (28x28 images flattened to 784).
    
    Args:
        input_size (int): Input dimension (default: 784 for MNIST)
        hidden_size (int): Hidden layer dimension (default: 100)
        output_size (int): Output dimension (default: 10 for 10 classes)
    """
    def __init__(
        self, 
        input_size=784, 
        hidden_size=100, 
        output_size=10
    ):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        total_weights = sum(p.numel() for p in self.parameters())
        print(f'Total number of weights: {total_weights}')  

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

#class Softmax_MaxCap(cnn.Module):
#    def __init__(self, dim=1, cap=100):
#        super().__init__()
#        self.dim = dim
#        self.cap = cap
#
#    def forward(self, x):
#        x = (x/self.cap)
#        x = x.tanh()*self.cap
#        logits = x - self.cap
#        numerator = logits.exp()
#        with cfg.temp_override({"functions.reciprocal_all_pos": True}):
#            inv_denominator = numerator.sum(self.dim, keepdim=True).reciprocal()
#        return numerator * inv_denominator


class CryptenThreeLayerNN(cnn.Module):
    """
    Three-layer feedforward neural network (CrypTen encrypted version).
    
    Same architecture as ThreeLayerNN but using CrypTen encrypted operations
    for secure multi-party computation training.
    
    All operations use crypten.nn modules for homomorphic encryption
    compatibility. Can be trained on encrypted data across multiple parties.
    
    Args:
        input_size (int): Input dimension (default: 784 for MNIST)
        hidden_size (int): Hidden layer dimension (default: 100)
        output_size (int): Output dimension (default: 10 for 10 classes)
    
    Note:
        Must be instantiated within @mpc.run_multiprocess() decorator
        for proper multi-party computation.
    """
    def __init__(self, 
        input_size=784, 
        hidden_size=100, 
        output_size=10
    ):
        super(CryptenThreeLayerNN, self).__init__()
        self.fc1 = cnn.Linear(input_size, hidden_size)
        self.fc2 = cnn.Linear(hidden_size, hidden_size)
        self.fc3 = cnn.Linear(hidden_size, output_size)
        self.relu = cnn.ReLU()
        # softmax approximated with 2 RELUs

        self.softmax = cnn.Softmax(dim=1)
        #self.softmax = cnn.SoftmaxMaxCap(dim=1, max_value=50)

        total_weights = sum(p.numel() for p in self.parameters())
        print(f'Total number of weights: {total_weights}')  
    
    def softmax_cap(self,x, dim=1):
        cap = 100
        x = (x/cap)
        x = x.tanh()*cap    
        logits = x - cap
        numerator = logits.exp()
        with cfg.temp_override({"functions.reciprocal_all_pos": True}):
            inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
        return numerator * inv_denominator
        
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x_pre_sax = x
        x = self.softmax(x)
        #dim = 1
        #cap = 100
        #x = (x/cap)
        #x = x.tanh()*cap    
        #logits = x - cap
        #numerator = logits.exp()
        #with cfg.temp_override({"functions.reciprocal_all_pos": True}):
        #    inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
        #x = numerator * inv_denominator

        x_post_sax = x.get_plain_text()
        if (x_post_sax.abs() > 10e5).any():
        #    if (x_pre_sax.abs() > 10e5).any():
        #        print("Overflow before SMAX")
        #    else:
            print("Overflow in SMAX")
            
        return x



class LeafCNN(nn.Module):
    def __init__(
        self, 
        input_channels: int = 1, 
        only_digits: bool = True
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(5,5), stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5,5), padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        flattened_size = 8*8*64 if input_channels == 3 else 7*7*64
        self.fc1 = nn.Linear(flattened_size, 2048)
        self.fc2 = nn.Linear(2048, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        self.input_channels = input_channels

    def forward(self, x):
        if x.shape[1] != self.input_channels:
            x = x.view(-1, self.input_channels, 28, 28)  # reshape input tensor
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
    

    
    
    
# Crypten equivalent of the Leaf model Net (hidden size for fc 2048)
# GOogle model has hidden size for fc 512
class CryptenLeafCNN(cnn.Module):
    def __init__(self, 
        input_channels: int = 1,
        only_digits: bool = True
    ):
        super().__init__()
        self.only_digits = only_digits
        self.conv1 = cnn.Conv2d(input_channels, 32, kernel_size=(5,5), stride=1, padding=2)
        self.pool1 = cnn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = cnn.Conv2d(32, 64, kernel_size=(5,5), padding=2)
        self.pool2 = cnn.MaxPool2d(kernel_size=2, stride=2)
        flattened_size = 8*8*64 if input_channels == 3 else 7*7*64
        self.fc1 = cnn.Linear(flattened_size, 2048)
        self.fc2 = cnn.Linear(2048, 10 if only_digits else 62)
        self.relu = cnn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits
    



class SmallCNN(nn.Module):
    def __init__(
        self, 
        input_channels: int = 1,
        only_digits: bool = True
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10 if only_digits else 62)  # Output size: 10 for digits, 62 for full dataset
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)  # Flatten tensor
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.softmax(self.fc2(x))  # Apply softmax to output
        return x

# Example Usage
# EMNIST 10 classes
#model_digits = SmallCNN(only_digits=True)
#total_weights = sum(p.numel() for p in model_digits.parameters())
#print(f'Total number of weights Small CNN (10 classes): {total_weights}')   

# FOR CIFAR-10
#model_cifar10 = SmallCNN(input_channels=3)
#total_weights = sum(p.numel() for p in model_cifar10.parameters())
#print(f'Total number of weights Small CNN (CIFAR-10): {total_weights}')

       