import os 

import numpy as np
import scipy.io
import skimage.io
import skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import to_tensor
from nn import get_random_batches

def get_labels(data, n):
    labels = np.zeros((data.shape[0], n))
    labels[np.arange(data.shape[0]), data[:].astype('int')] = 1
    return labels

def data_loader(path, n, offset=1):
    data, labels = [], []
    for root, dirs, files in os.walk(path, topdown=False):
        print(root)
        if root.split('/')[-1] == path.split('/')[-1]: continue
        data += [ skimage.transform.resize(skimage.io.imread(os.path.join(root, filename)).astype('float')/255, (224, 224, 3)) for filename in files ]
        labels += [ int(root.split('/')[-1]) - offset ] * len(files)

    data = np.transpose(data, (0, 3, 1, 2))
    labels = get_labels(np.asarray(labels), n)
    indices = np.random.permutation(data.shape[0])
    return data[indices], labels[indices]

def reformat_data(n):
    train_x, train_y = data_loader('../data/oxford-flowers%d/train'%n, n)
    valid_x, valid_y = data_loader('../data/oxford-flowers%d/val'%n, n)
    scipy.io.savemat('../data/flowers%d_train.mat'%n, {'train_data': train_x, 'train_labels': train_y})
    scipy.io.savemat('../data/flowers%d_valid.mat'%n, {'valid_data': valid_x, 'valid_labels': valid_y})

# Load data
n = 17
train_data = scipy.io.loadmat('../data/flowers%d_train.mat'%n)
valid_data = scipy.io.loadmat('../data/flowers%d_valid.mat'%n)
train_x, train_y = to_tensor(train_data['train_data']), to_tensor(train_data['train_labels'])
valid_x, valid_y = to_tensor(valid_data['valid_data']), to_tensor(valid_data['valid_labels'])

# Hyperparameters
max_iters = 70
learning_rate = 1e-2
momentum = 0.9
batch_size = 50
train_batches = get_random_batches(train_x, train_y, batch_size)
valid_batches = get_random_batches(valid_x, valid_y, batch_size)

# Network model
model_name = 'lenet5_flowers%d'%n
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 60, kernel_size=5)
        self.fc1 = torch.nn.Linear(60*53*53, 120)
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
