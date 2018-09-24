import os
import time

import numpy as np
import scipy.ndimage

import util

def extract_deep_feature(x, vgg16_weights):
    '''
    Extracts deep features from the given VGG-16 weights.

    [input]
    * x: numpy.ndarray of shape (H, W, 3)
    * vgg16_weights: numpy.ndarray of shape (L, 3)

    [output]
    * feat: numpy.ndarray of shape (K)
    '''

    ops = {'conv2d': multichannel_conv2d, 'relu': relu,
           'maxpool2d': max_pool2d, 'linear': linear}

    # Pass image through the different layers of VGG-16
    for layer in vgg16_weights[:35]:
        print('\n', layer[0], '-'*30)
        print('Input:', x.shape)

        args = [x] + layer[1:] if (len(layer) > 1) else [x]
        x = ops[layer[0]](*args)

        print('Output:', x.shape)

    np.save('deep_features', x)
    print(x.shape)
    return x

def multichannel_conv2d(x, weight, bias):
    '''
    Performs multi-channel 2D convolution.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * weight: numpy.ndarray of shape (output_dim, input_dim, kernel_size, kernel_size)
    * bias: numpy.ndarray of shape (output_dim)

    [output]
    * feat: numpy.ndarray of shape (H, W, output_dim)
    '''

    y = []
    for i in range(weight.shape[0]):
        H = []
        for j in range(weight.shape[1]):
            # Flip LR the weight matrix
            W = np.fliplr(weight[i, j, :, :])
            # 2D convolve on each input channel
            H.append(scipy.ndimage.convolve(x[:, :, j], W))
        # Sum up all the convulution responses and the bias term
        y.append(sum(H) + bias[i])

    # Stack all filter responses back
    y = np.dstack(y)
    return y

def relu(x):
    '''
    Rectified linear unit.

    [input]
    * x: numpy.ndarray

    [output]
    * y: numpy.ndarray
    '''
    
    y = np.maximum(x, 0)
    return y

def max_pool2d(x, size):
    '''
    2D max pooling operation.

    [input]
    * x: numpy.ndarray of shape (H, W, input_dim)
    * size: pooling receptive field

    [output]
    * y: numpy.ndarray of shape (H/size, W/size, input_dim)
    '''
    
    # Make the input divisible by the pooling size
    FH = (x.shape[0]//size) * size
    FW = (x.shape[1]//size) * size

    # Max pool on each channel
    y = []
    for i in range(x.shape[2]):
        # Reshape the array into groups of pooling size
        temp = x[:FH, :FW, i].reshape(FH//size, size, FW//size, size)
        # Pick the maximum value from each (size x size) block
        y.append(temp.max(axis=(1, 3)))

    # Stack all responses back
    y = np.dstack(y)
    return y

def linear(x, W, b):
    '''
    Fully-connected layer.

    [input]
    * x: numpy.ndarray of shape (input_dim)
    * W: numpy.ndarray of shape (output_dim, input_dim)
    * b: numpy.ndarray of shape (output_dim)

    [output]
    * y: numpy.ndarray of shape (output_dim)
    '''
    
    y = np.dot(W, x.flatten()) + b
    return y
