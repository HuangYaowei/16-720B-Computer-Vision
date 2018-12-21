#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 5 - Neural Networks for Recognition
'''

__author__ = "Heethesh Vhavle"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"
__email__ = "heethesh@cmu.edu"

import os

import numpy as np
import scipy.io
import skimage
import skimage.io
import skimage.util
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches

import torch
import torch.optim as optim
import torch.nn.functional as F

from q4 import *
from cnn_emnist import *
from util import to_tensor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

ground_thruth = [list('TODOLIST1MAKEATODOLIST2CHECKOFFTHEFIRSTTHINGONTODOLIST3REALIZEYOUHAVEALREADYCOMPLETED2THINGS4REWARDYOURSELFWITHANAP'),
                 list('ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'),
                 list('HAIKUSAREEASYBUTSOMETIMESTHEYDONTMAKESENSEREFRIGERATOR'),
                 list('DEEPLEARNINGDEEPERLEARNINGDEEPESTLEARNING')]

for img in os.listdir('../images'):
    # Read the test image
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images', img)))
    print('\nDetecting >>>', img)
    
    # Find and sort letters per line
    bboxes, bw = findLetters(im1)
    bboxes, counts = sortLetters(bboxes)
    # plt.imshow(bw)

    # Preprocessing
    letters_to_detect = []
    for i, bbox in enumerate(bboxes):
        minr, minc, maxr, maxc = bbox

        # Invert, rescale and crop letters
        crop = skimage.util.invert(bw[minr:maxr, minc:maxc])
        crop = skimage.transform.rescale(crop, 20/max(maxc-minc, maxr-minr))
        
        # Pad letters with odd shape offsets
        ypad, xpad = (28 - crop.shape[0])//2, (28 - crop.shape[1])//2
        yodd, xodd = int(crop.shape[0]%2!=0), int(crop.shape[1]%2!=0)
        letter = np.pad(crop, ((ypad, ypad + 1*yodd), (xpad, xpad + 1*xodd)), 'constant', constant_values=(0, 0))
        skimage.io.imsave('../crops_emnist/%s_%02d.png'%(img.split('.')[0], i), letter)

        # Transpose and add letter for detection
        letters_to_detect.append(letter.T)

        # Draw bounding boxes
        rect = matplotlib.patches.Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='red', linewidth=2)
        # plt.gca().add_patch(rect)
    
    # Display the image
    # plt.show()

    # Test data
    letters_to_detect = to_tensor(np.vstack(letters_to_detect).reshape(len(letters_to_detect), 1, 28, 28))
    
    # Load model and weights
    net = Net()
    net.load_state_dict(torch.load('weights/cnn_emnist.pt'))
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict) 
    # model.load_state_dict(model_dict)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    letters_to_detect = letters_to_detect.to(device)

    # Detect the letters using the neural network
    probs = net(letters_to_detect).cpu().detach().numpy()
    detects = np.argmax(probs, axis=1)

    # Infer the detections
    letters = np.char.asarray([ chr(j) for i, j in scipy.io.loadmat('../data/emnist_mapping.mat')['mapping'].tolist() ])
    lines = np.split(letters[detects], counts[:-1])
    lines = [ ''.join(line.tolist()) for line in lines ]

    # Calculate accuracy
    correct = (np.char.asarray(ground_thruth[int(img[1])-1]) == letters[detects])
    print('Actual Accuracy: %.2f%%'%(correct.sum()*100.0/correct.size))
    correct = (np.char.asarray(ground_thruth[int(img[1])-1]).lower() == letters[detects].lower())
    print('Absolute Accuracy: %.2f%%\n'%(correct.sum()*100.0/correct.size))

    # Display the detected lines
    for line in lines: print(line)
