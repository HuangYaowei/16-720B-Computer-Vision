import scipy.io
import torch
import torch.nn as nn

from util import to_tensor
from nn import get_random_batches

# Load data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
train_x, train_y = to_tensor(train_data['train_data']), to_tensor(train_data['train_labels'])
valid_x, valid_y = to_tensor(valid_data['valid_data']), to_tensor(valid_data['valid_labels'])

# Hyperparameters
max_iters = 100
learning_rate = 1e-2
momentum = 0.9
hidden_size = 64
batch_size = 50
batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

# Network model
model_name = 'fcn_nist36'
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(train_x.shape[1], hidden_size)
        self.output = nn.Linear(hidden_size, train_y.shape[1])

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = self.output(x)
        return x
