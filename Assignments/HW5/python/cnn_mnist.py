#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 5 - Neural Networks for Recognition
'''

__author__ = "Heethesh Vhavle"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"
__email__ = "heethesh@cmu.edu"

import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import to_tensor
from nn import get_random_batches

def get_labels(data):
    labels = np.zeros((data.shape[0], 10))
    labels[np.arange(data.shape[0]), data[:, -1].astype('int')] = 1
    return labels

def save_mat(split, data):
    mat = {split + '_data': data[:, :-1], split + '_labels': get_labels(data)}
    scipy.io.savemat('../data/mnist_' + split, mat)

def reformat_data():
    data = np.loadtxt('../data/mnist_test.txt')
    save_mat('valid', data)

# Load data (Source: http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/)
train_data = scipy.io.loadmat('../data/mnist_train.mat')
valid_data = scipy.io.loadmat('../data/mnist_valid.mat')
train_x, train_y = train_data['train_data'], to_tensor(train_data['train_labels'])
valid_x, valid_y = valid_data['valid_data'], to_tensor(valid_data['valid_labels'])

# Reshape data
train_x = to_tensor(train_x.reshape(train_x.shape[0], 1, 28, 28))
valid_x = to_tensor(valid_x.reshape(valid_x.shape[0], 1, 28, 28))

# Hyperparameters
max_iters = 100
learning_rate = 1e-2
momentum = 0.9
batch_size = 50
train_batches = get_random_batches(train_x, train_y, batch_size)
valid_batches = get_random_batches(valid_x, valid_y, batch_size)

# Network model
model_name = 'cnn_mnist'
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16*4*4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, train_y.shape[1])

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
