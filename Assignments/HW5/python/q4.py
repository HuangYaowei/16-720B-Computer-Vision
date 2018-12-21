#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 5 - Neural Networks for Recognition
'''

__author__ = "Heethesh Vhavle"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"
__email__ = "heethesh@cmu.edu"

import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.color
import skimage.filters
import skimage.morphology
import skimage.segmentation

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

BLUR_SIGMA = 2
BW_CLOSING_ITERS = 3
MIN_BBOX_AREA = 1000
LABELS_EROSION_ITERS = 4
JOINT_MIN_ASPECT_RATIO = 1.15
JOINT_EROSION_ITERS = 1
LINE_DETECTION_THRESH = 100

def findLabels(image, morph_iters=LABELS_EROSION_ITERS):
    for _ in range(morph_iters): image = skimage.morphology.binary_erosion(image)
    image = skimage.img_as_float(image)
    labels = skimage.measure.label(image-1)
    labels = skimage.segmentation.clear_border(labels)
    regions = skimage.measure.regionprops(labels)
    return labels, regions

def drawBoxes(image, bboxes):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = mpatches.Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

def removeJoints(image, bboxes):
    i = 0
    while i < len(bboxes):
        minr, minc, maxr, maxc = bboxes[i]
        if (maxc - minc) / (maxr - minr) > JOINT_MIN_ASPECT_RATIO:
            letter = np.ones_like(image)
            letter[minr:maxr, minc:maxc] = image[minr:maxr, minc:maxc]
            _, regions = findLabels(letter, morph_iters=JOINT_EROSION_ITERS)
            n = len(regions)
            if n > 1:
                print('Found %d joined letters'%n)
                bboxes = bboxes[:i] + [ region.bbox for region in regions ] + bboxes[i+1:]
                i += (n - 1)
        i += 1

    return bboxes

def sortLetters(bboxes):
    # Sort all letters row wise
    bboxes.sort(key=lambda x: (x[0]))
    
    # Find lines and group the letters per line
    lines, words = [], []
    last_line = 1e8
    for bbox in bboxes:
        if abs(last_line - bbox[0]) > LINE_DETECTION_THRESH:
            if words: lines.append(words)
            words = []
        words.append(bbox)
        last_line = bbox[0]
    if words: lines.append(words)

    # Sort the letters column wise per line
    counts = []
    bboxes_sorted = []
    for line in lines:
        line.sort(key=lambda x: (x[1]))
        counts.append(len(line))
        bboxes_sorted += line

    return bboxes_sorted, np.cumsum(counts)
    
def findLetters(image):
    # Preprocess and threshold the image
    image = skimage.img_as_float(image)
    image = skimage.color.rgb2gray(image)
    image = image / np.max(image)
    image = skimage.filters.gaussian(image, sigma=BLUR_SIGMA)
    thresh = skimage.filters.threshold_otsu(image)
    bw = image > thresh

    # Apply morphological opertaions
    iters = (image.shape[0] * image.shape[1])//int(1.5e6) + 2*int(image.shape[1]>4000)
    for _ in range(iters): bw = skimage.morphology.binary_erosion(bw)
    for _ in range(BW_CLOSING_ITERS): bw = skimage.morphology.binary_closing(bw)
    bw = skimage.img_as_float(bw)

    # Assign labels and find the bounding boxes
    labels, regions = findLabels(bw)
    bboxes = [ region.bbox for region in regions if region.area > MIN_BBOX_AREA]
    bboxes = removeJoints(bw, bboxes)
    if __name__ == '__main__': drawBoxes(bw, bboxes)
    
    return bboxes, bw

if __name__ == '__main__':
    n = 1
    # n = np.random.randint(1, 5)
    if n==1: img = skimage.io.imread('../images/01_list.jpg')
    if n==2: img = skimage.io.imread('../images/02_letters.jpg')
    if n==3: img = skimage.io.imread('../images/03_haiku.jpg')
    if n==4: img = skimage.io.imread('../images/04_deep.jpg')
    findLetters(img)
