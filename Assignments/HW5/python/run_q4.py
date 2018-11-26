import os
import pickle
import string

import numpy as np
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches

from nn import *
from q4 import *

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

        # Rescale and crop letters
        crop = bw[minr:maxr, minc:maxc]
        crop = skimage.transform.rescale(crop, 26/max(maxc-minc, maxr-minr))
        
        # Pad letters with odd shape offsets
        ypad, xpad = (32 - crop.shape[0])//2, (32 - crop.shape[1])//2
        yodd, xodd = int(crop.shape[0]%2!=0), int(crop.shape[1]%2!=0)
        letter = np.pad(crop, ((ypad, ypad + 1*yodd), (xpad, xpad + 1*xodd)), 'constant', constant_values=(1, 1))
        skimage.io.imsave('../crops/%s_%02d.png'%(img.split('.')[0], i), letter)

        # Transpose and add letter for detection
        letters_to_detect.append(letter.T.flatten())

        # Draw bounding boxes
        rect = matplotlib.patches.Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='red', linewidth=2)
        # plt.gca().add_patch(rect)
    
    # Display the image
    # plt.show()

    # Test data
    letters_to_detect = np.vstack(letters_to_detect)
    
    # Load the weights
    params = pickle.load(open('q3_weights.pickle', 'rb'))
    
    # Detect the letters using the neural network
    h1 = forward(letters_to_detect, params, name='layer1', activation=sigmoid)
    probs = forward(h1, params, name='output', activation=softmax)
    detects = np.argmax(probs, axis=1)

    # Infer the detections
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    lines = np.split(letters[detects], counts[:-1])
    lines = [ ''.join(line.tolist()) for line in lines ]

    # Calculate accuracy 
    correct = (ground_thruth[int(img[1])-1] == letters[detects])
    print('Accuracy: %.2f%%\n'%(correct.sum()*100.0/correct.size))

    # Display the detected lines
    for line in lines: print(line)
