import sys
import pickle
import string

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from nn import *

# Load data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

# Hyperparameters
max_iters = 100
learning_rate = 1e-2
hidden_size = 64
batch_size = 50
batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

params = {}

# Initialize layers
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, train_y.shape[1], params, 'output')

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
        delta = probs - yb
        delta1 = backwards(delta, params, 'output', linear_deriv)
        delta2 = backwards(delta1, params, 'layer1', sigmoid_deriv)

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

# Run on validation set and report accuracy! Should be above 75%
h1 = forward(valid_x, params, 'layer1', sigmoid)
probs = forward(h1, params, 'output', softmax)
loss, valid_acc = compute_loss_and_acc(valid_y, probs)
print('\nValidation Accuracy:', valid_acc)

if False:
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32, 32).T)
        plt.show()

saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights_new.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
sys.exit(0)

# Q3.1.3

# Q3.1.3
confusion_matrix = np.zeros((train_y.shape[1], train_y.shape[1]))
plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
