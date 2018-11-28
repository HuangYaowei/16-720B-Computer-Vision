import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

from nn import *

# Load data
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

# Hyperparameters
max_iters = 2
learning_rate = 3e-5
momentum = 0.9
lr_rate = 20
hidden_size = 32
batch_size = 50
batches = get_random_batches(train_x, np.ones((train_x.shape[0], 1)), batch_size)
batch_num = len(batches)

params = {}

# Initialize layers here
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'hidden')
initialize_weights(hidden_size, hidden_size, params, 'hidden2')
initialize_weights(hidden_size, train_x.shape[1], params, 'output')

# Initialize momentums
for layer in ['output', 'layer1', 'hidden', 'hidden2']:
    params['m_' + layer] = np.zeros_like(params['W' + layer])

# Training loop
all_loss = []
for itr in range(max_iters):
    total_loss = 0
    for xb, _ in batches:
        # Forward pass
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'hidden', relu)
        h3 = forward(h2, params, 'hidden2', relu)
        out = forward(h3, params, 'output', sigmoid)

        # Loss function
        error = xb - out
        total_loss += np.sum(error**2)

        # Backward pass
        delta = -2 * error
        delta1 = backwards(delta, params, 'output', sigmoid_deriv)
        delta2 = backwards(delta1, params, 'hidden2', relu_deriv)
        delta3 = backwards(delta2, params, 'hidden', relu_deriv)
        delta4 = backwards(delta3, params, 'layer1', relu_deriv)

        # Apply gradient
        for layer in ['output', 'layer1', 'hidden', 'hidden2']:
            params['m_' + layer] = momentum*params['m_' + layer] - learning_rate*params['grad_W' + layer] 
            params['W' + layer] += params['m_' + layer]
            params['b' + layer] -= learning_rate * params['grad_b' + layer]
    
    # Save loss
    all_loss.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} lr: {:.6f} loss: {:.2f}".format(itr, learning_rate, total_loss))
    
    if itr % lr_rate == lr_rate-1:
        learning_rate *= momentum

# Q5.3.1
indices = [0, 1, 1000, 1001, 3000, 3001, 5000, 5001, 7000, 7001]
h1 = forward(train_x[indices], params, 'layer1', relu)
h2 = forward(h1, params, 'hidden', relu)
h3 = forward(h2, params, 'hidden2', relu)
out = forward(h3, params, 'output', sigmoid)

# Visualize some results
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(0, 10, 2):
    plt.subplot(2, 2, 1)
    plt.imshow(train_x[indices[i]].reshape(32, 32).T, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(out[i].reshape(32, 32).T, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(train_x[indices[i+1]].reshape(32, 32).T, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(out[i+1].reshape(32, 32).T, cmap='gray')
    plt.show()

# Plot epoch vs loss
plt.plot(np.arange(max_iters), all_loss)
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Total Loss')
plt.show()

# Q5.3.2
psnrs = 0
for xb in valid_x:
    # Forward pass
    h1 = forward(xb, params, 'layer1', relu)
    h2 = forward(h1, params, 'hidden', relu)
    h3 = forward(h2, params, 'hidden2', relu)
    out = forward(h3, params, 'output', sigmoid)
    
    # Evaluate PSNR
    psnrs += psnr(xb, out)

# Average PSNR
print(psnrs/valid_x.shape[0])
