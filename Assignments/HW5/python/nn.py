import numpy as np
from util import *

'''
Q 2.1
Initialize b to 0 vector
b should be a 1D array, not a 2D array with a singleton dimension
We will do XW + b. 
X be [Examples, Dimensions]
'''
def initialize_weights(in_size, out_size, params, name=''):
    # Xavier initialization
    var = 2 / (in_size + out_size)
    std = np.sqrt(3 * var)

    W = np.random.uniform(-std, std, (in_size, out_size))
    b = np.zeros(out_size)

    params['W' + name] = W
    params['b' + name] = b

'''
Q 2.2.1
x is a matrix
A sigmoid activation function
'''
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

'''
Q 2.2.1
Get the layer parameters
Do a forward pass

Keyword arguments:
X - input vector [Examples x D]
params - a dictionary containing parameters
name - name of the layer
activation - the activation function (default is sigmoid)
'''
def forward(X, params, name='', activation=sigmoid):
    W = params['W' + name]
    b = params['b' + name]

    # Forward pass
    pre_act = np.dot(X, W) + b
    post_act = activation(pre_act)

    # Store the pre-activation and post-activation values
    # These will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

'''
Q 2.2.2 
x is [examples, classes]
Softmax should be done for each row
'''
def softmax(x):
    if x.shape[0] == 1:
        res = np.exp(x - np.amax(x))
        res = res / np.sum(res)
    else:
        res = np.exp(x - np.amax(x, axis=1).reshape(-1, 1))
        res = res / np.sum(res, axis=1).reshape(-1, 1)
    return res

'''
Q 2.2.3
Compute total loss and accuracy
y is size [examples, classes]
probs is size [examples, classes]
'''
def compute_loss_and_acc(y, probs):
    loss = -np.sum(y * np.log(probs))
    acc = np.mean(np.sum(y * np.where(probs==probs.max(axis=1).reshape(-1, 1), 1, 0), axis=1))
    return loss, acc

'''
We give this to you because you proved it
It's a function of post_act
'''
def sigmoid_deriv(post_act):
    res = post_act * (1.0 - post_act)
    return res

'''
Q 2.3.1
Do a backwards pass
Keyword arguments:
delta - errors to backprop
params - a dictionary containing parameters
name - name of the layer
activation_deriv - the derivative of the activation_func
'''
def backwards(delta, params, name='', activation_deriv=sigmoid_deriv):
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    
    # Do the derivative through activation first
    # Then compute the derivative W, b and X
    grad_A = delta * activation_deriv(post_act)
    grad_X = grad_A @ W.T
    grad_W = X.T @ grad_A
    grad_b = np.full(b.shape, np.sum(grad_A))

    # Store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

'''
Q 2.4
Split x and y into random batches
Return a list of [(batch1_x, batch1_y), ...]
'''
def get_random_batches(x, y, batch_size):
    random_batches = np.array_split(np.random.permutation(x.shape[0]), x.shape[0]//batch_size)
    batches = [ (x[batch], y[batch]) for batch in random_batches ]
    return batches
