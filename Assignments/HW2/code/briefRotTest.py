import cv2
import numpy as np
import matplotlib.pyplot as plt

from BRIEF import *

def rotate_image(im, degrees):
    # Convert RGB to grayscale
    # if len(im.shape) == 3:
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    rows, cols, c = im.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degrees, 1)
    dst = cv2.warpAffine(im, M, (cols, rows))

    return dst

def display_histogram(hist, bins=100):
    '''
    Displays a numpy histogram

    [input]
    * hist: a numpy.ndarray of shape (K)
    * bins: int of size (K)
    '''

    fig, ax = plt.subplots()
    plt.bar(range(bins), hist)
    ax.set_xticks(range(bins))
    ax.set_xticklabels([ str(i*10) for i in range(bins) ])
    plt.xlabel('Rotation Angle')
    plt.ylabel('Number of Matches')
    plt.show()

def plot_histogram(im):
    # Get descriptors for original image
    locs1, desc1 = briefLite(img)
    
    count = []
    for i in range(36):
        # Get descriptors for rotated image
        rot_img = rotate_image(img, i*10)
        locs2, desc2 = briefLite(rot_img)

        # Find matches
        matches = briefMatch(desc1, desc2)
        count.append(len(matches))
        # plotMatches(img, rot_img, matches, locs1, locs2)

    display_histogram(count, 36)

if __name__ == '__main__':
    # Load image
    img = cv2.imread('../data/model_chickenbroth.jpg')
    plot_histogram(img)
