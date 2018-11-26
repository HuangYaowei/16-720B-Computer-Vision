import torch
import torch.optim as optim
import torch.nn.functional as F

def accuracy(probs, y):
    probs = F.softmax(probs, dim=1)
    correct = (torch.max(probs, 1)[1] == torch.max(y, 1)[1]).sum()
    acc = correct.float() / y.size()[0]
    return acc.item()

def train():
    # Network model
    net = Net()

    # Optimizer function
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    
    # Training loop
    all_loss, all_acc = [], []
    for itr in range(max_iters):
        total_loss, total_acc = 0, 0
        for xb, yb in batches:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = net(xb)
            loss = F.cross_entropy(outputs, torch.max(yb, 1)[1])
            loss.backward()
            optimizer.step()

            # Loss and accuracy
            total_loss += loss.item()
            total_acc += accuracy(outputs, yb)

        # Total accuracy
        avg_acc = total_acc / batch_num

        # Validation forward pass, loss and accuracy
        voutputs = net(valid_x)
        valid_loss = F.cross_entropy(voutputs, torch.max(valid_y, 1)[1])
        valid_acc = accuracy(voutputs, valid_y)

        # Save for plotting
        all_loss.append([total_loss, valid_loss])
        all_acc.append([avg_acc, valid_acc])

        if itr % 2 == 0:
            print("itr: {:03d} loss: {:.2f} acc: {:.2f} vloss: {:.2f} vacc: {:.2f}".format(itr, total_loss, avg_acc, valid_loss, valid_acc))

    return all_loss, all_acc

if __name__ == '__main__':
    # Select model
    # from fcn_nist36 import *
    from cnn_mnist import *
    # from cnn_nist36 import *
    # from cnn_emnist import *

    # Training loop
    train()
    