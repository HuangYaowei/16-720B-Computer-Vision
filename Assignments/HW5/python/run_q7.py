#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 5 - Neural Networks for Recognition
'''

__author__ = "Heethesh Vhavle"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"
__email__ = "heethesh@cmu.edu"

import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from util import plot

def accuracy(probs, y):
    probs = F.softmax(probs, dim=1)
    correct = (torch.max(probs, 1)[1] == torch.max(y, 1)[1]).sum()
    acc = correct.float() / y.size(0)
    return acc.item()

def train(save=False):
    # Optimizer function
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    
    # Training loop
    all_loss, all_acc = [], []
    for itr in range(max_iters):
        # Train
        net.train()
        train_loss, train_acc = 0, 0
        for xb, yb in train_batches:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = net(xb)
            loss = F.cross_entropy(outputs, torch.max(yb, 1)[1])
            loss.backward()
            optimizer.step()

            # Loss and accuracy
            train_loss += loss.item()
            train_acc += accuracy(outputs, yb)

        # Validate
        net.eval()
        valid_loss, valid_acc = 0, 0
        for xb, yb in valid_batches:
            outputs = net(xb)
            valid_loss += F.cross_entropy(outputs, torch.max(yb, 1)[1]).item()
            valid_acc += accuracy(outputs, yb)
        
        # Total accuracy
        train_acc = train_acc / len(train_batches)
        valid_acc = valid_acc / len(valid_batches)

        # Save for plotting
        all_loss.append([train_loss, valid_loss])
        all_acc.append([train_acc, valid_acc])

        if itr % 1 == 0:
            print("{:s} | itr: {:03d} | loss: {:.2f} | acc: {:.2f} | vloss: {:.2f} | vacc: {:.2f}".format(model_name, itr, train_loss, train_acc, valid_loss, valid_acc))

    # Save the learned parameters and graphs
    if save: 
        torch.save(net.state_dict(), 'weights/%s.pt'%model_name)
        np.savez('weights/%s_graphs'%model_name, loss=np.asarray(all_loss), accuracy=np.asarray(all_acc))
    
    return np.asarray(all_loss), np.asarray(all_acc)

if __name__ == '__main__':
    # Select model
    mode = 4
    if mode==1: from fcn_nist36 import *
    if mode==2: from cnn_mnist import *
    if mode==3: from cnn_nist36 import *
    if mode==4: from cnn_emnist import *
    if mode==5: from lenet5_flowers import *
    if mode==0: from squeezenet_flowers import *

    # Network model
    if mode: net = Net()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    if torch.cuda.is_available(): 
        train_batches = [ (x.to(device), y.to(device)) for x, y in train_batches ]
        valid_batches = [ (x.to(device), y.to(device)) for x, y in valid_batches ]

    # Training loop
    start = time.time()
    loss, acc = train(save=True)
    print('Took %.2fs'%(time.time() - start))

    # Plot and save graphs
    plot(loss, 'Loss', model_name, max_iters)
    plot(acc, 'Accuracy', model_name, max_iters)
    