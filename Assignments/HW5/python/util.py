import torch
import numpy as np

# use for a "no activation" layer
def linear(x):
    return x

def linear_deriv(post_act):
    return np.ones_like(post_act)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(post_act):
    return 1 - post_act**2

def relu(x):
    return np.maximum(x, 0)

def relu_deriv(x):
    return (x > 0).astype(np.float)

def to_tensor(array):
    tensor = torch.from_numpy(array).type(torch.FloatTensor)
    return tensor

def plot(data, title, model_name, max_iters):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(max_iters), data[:, 0])
    plt.plot(np.arange(max_iters), data[:, 1])
    plt.title('%s Curve'%title)
    plt.xlabel('Epochs')
    plt.ylabel('Total %s'%title)
    plt.legend(['Train Data', 'Validation Data'])
    plt.grid()
    plt.savefig('../writeup/plots/%s_%s.png'%(model_name, title.lower()))
    plt.show()
