#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 5 - Neural Networks for Recognition
'''

__author__ = "Heethesh Vhavle"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"
__email__ = "heethesh@cmu.edu"

import pickle
import string

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from nn import *
from util import plot
from grid import display_grid

TRAIN = True
TEST = True
SAVE_PARAMS = False
LOAD_PARAMS = False
model_name = 'q3_fcn_nist36'

def visualize_weights(weights):
    weights_list = [ weight.reshape(32, 32) for weight in weights.T ]
    display_grid(weights_list, (8, 8), axes_pad=0.05, cmap='binary')
    
# Load data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

if TRAIN:
    # Hyperparameters
    max_iters = 100
    learning_rate = 1e-3
    hidden_size = 64
    batch_size = 50
    batches = get_random_batches(train_x, train_y, batch_size)
    batch_num = len(batches)
    
    # Training parameters
    params = {}

    # Initialize layers
    initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
    initialize_weights(hidden_size, train_y.shape[1], params, 'output')

    # Visualize weights
    visualize_weights(params['Wlayer1'])

    # Training loop
    # With default settings, you should get loss < 150 and accuracy > 80%
    all_loss, all_acc = [], []
    for itr in range(max_iters):
        total_loss, total_acc = 0, 0
        for xb, yb in batches:
            # Forward pass
            h1 = forward(xb, params, 'layer1', sigmoid)
            probs = forward(h1, params, 'output', softmax)

            # Loss and accuracy
            loss, acc = compute_loss_and_acc(yb, probs)
            total_loss += loss
            total_acc += acc

            # Backward pass
            delta1 = probs
            delta1[np.arange(probs.shape[0]), np.argmax(yb, axis=1)] -= 1
            delta2 = backwards(delta1, params, 'output', linear_deriv)
            delta3 = backwards(delta2, params, 'layer1', sigmoid_deriv)

            # Apply gradient
            for layer in ['output', 'layer1']:
                params['W' + layer] -= learning_rate * params['grad_W' + layer]
                params['b' + layer] -= learning_rate * params['grad_b' + layer]
        
        # Average loss and accuracy
        avg_acc = total_acc / batch_num
        total_loss = total_loss / batch_num

        # Validation forward pass
        vh1 = forward(valid_x, params, 'layer1', sigmoid)
        vprobs = forward(vh1, params, 'output', softmax)

        # Validation loss and accuracy
        valid_loss, valid_acc = compute_loss_and_acc(valid_y, vprobs)
        valid_loss /= valid_y.shape[0]
            
        # Save for plotting
        all_loss.append([total_loss, valid_loss])
        all_acc.append([avg_acc, valid_acc])

        if itr % 2 == 0:
            print("itr: {:03d} loss: {:.2f} acc: {:.2f} vloss: {:.2f} vacc: {:.2f}".format(itr, total_loss, avg_acc, valid_loss, valid_acc))

    print('\nTrain Accuracy:', all_acc[-1][0])

    # Run on validation set and report accuracy! Should be above 75%
    h1 = forward(valid_x, params, 'layer1', sigmoid)
    probs = forward(h1, params, 'output', softmax)
    loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    print('Validation Accuracy:', valid_acc)

    # Save trained weights
    if SAVE_PARAMS:
        saved_params = {k:v for k,v in params.items() if '_' not in k}
        with open('q3_weights.pickle', 'wb') as handle:
            pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot and save graphs
    plot(np.asarray(all_loss), 'Loss', model_name, max_iters)
    plot(np.asarray(all_acc), 'Accuracy', model_name, max_iters)

if TEST:
    # Load the weights
    if LOAD_PARAMS: params = pickle.load(open('q3_weights.pickle', 'rb'))

    # Test data forward pass
    h1 = forward(test_x, params, 'layer1', sigmoid)
    probs = forward(h1, params, 'output', softmax)
    test_loss, test_acc = compute_loss_and_acc(test_y, probs)
    test_loss /= test_y.shape[0]
    print('Test Accuracy:', test_acc)

    # Visualize weights
    visualize_weights(params['Wlayer1'])

    # Q3.1.3 - Compute confusion matrix
    confusion_matrix = np.zeros((test_y.shape[1], test_y.shape[1]))
    for true, pred in zip(np.argmax(test_y, axis=1).astype('int'), np.argmax(probs, axis=1).astype('int')):
        confusion_matrix[true, pred] += 1

    # Display confusion matrix
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='binary')
    plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
    plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
    plt.title('Confusion Matrix for NIST36 Test Data')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.colorbar()
    plt.grid()
    plt.show()
