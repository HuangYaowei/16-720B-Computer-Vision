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
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')
train_x = train_data['train_data']
valid_x = valid_data['valid_data']
test_x = test_data['test_data']

dim = 32

# Standardize data
# train_x -= np.mean(train_x)
# valid_x -= np.mean(valid_x)

# Do PCA
U, S, V = np.linalg.svd(train_x, full_matrices=False)

# Rebuild a low-rank version
lrank = V[:dim, :] 
print(lrank.shape)
print(np.linalg.matrix_rank(lrank))

# Rebuild it
recon = U[:, :dim] @ np.diag(S)[:dim, :dim] @ lrank
# for i in range(5):
#     plt.subplot(2,1,1)
#     plt.imshow(train_x[i].reshape(32,32).T)
#     plt.subplot(2,1,2)
#     plt.imshow(recon[i].reshape(32,32).T)
#     plt.show()

# Build valid dataset
recon_test = test_x @ lrank.T @ lrank

# Visualize results on test dataset
indices = [0, 1, 300, 301, 1000, 1001, 1300, 1301, 1700, 1701]
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(0, 10, 2):
    plt.subplot(2, 2, 1)
    plt.imshow(test_x[indices[i]].reshape(32, 32).T, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(recon_test[indices[i]].reshape(32, 32).T, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(test_x[indices[i+1]].reshape(32, 32).T, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(recon_test[indices[i+1]].reshape(32, 32).T, cmap='gray')
    plt.show()

# Build valid dataset
recon_valid = valid_x @ lrank.T @ lrank

total = []
for pred, gt in zip(recon_valid, valid_x):
    total.append(psnr(gt, pred))
print(np.array(total).mean())
