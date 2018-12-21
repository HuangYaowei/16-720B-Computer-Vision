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
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets

from util import to_tensor
from nn import get_random_batches

# Load data
NUM_CLASSES = 17
train_data = scipy.io.loadmat('../data/flowers17_train.mat')
valid_data = scipy.io.loadmat('../data/flowers17_valid.mat')
train_x, train_y = to_tensor(train_data['train_data']), to_tensor(train_data['train_labels'])
valid_x, valid_y = to_tensor(valid_data['valid_data']), to_tensor(valid_data['valid_labels'])

# Hyperparameters
max_iters = 100
learning_rate = 1e-2
momentum = 0.9
batch_size = 50
train_batches = get_random_batches(train_x, train_y, batch_size)
valid_batches = get_random_batches(valid_x, valid_y, batch_size)

# Load pretrained model
model_name = 'squeezenet'
net = models.squeezenet1_1(pretrained=True)

# Disable feature extraction
for param in net.parameters(): param.requires_grad = False

# Reshape the classifier
net.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
net.num_classes = NUM_CLASSES
