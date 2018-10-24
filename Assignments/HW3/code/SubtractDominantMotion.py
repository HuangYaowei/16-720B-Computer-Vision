#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 3 - Lucas-Kanade Tracking and Correlation Filters
'''

__author__ = "Heethesh Vhavle"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"
__email__ = "heethesh@cmu.edu"

import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage.morphology import binary_opening, binary_closing

from LucasKanadeAffine import LucasKanadeAffine

def SubtractDominantMotion(image1, image2, LK_method=LucasKanadeAffine, threshold=0.1, iters=7):
    '''
    [input]
    * image1 - Image at time t
    * image2 - Image at time t+1'
    * LK_method - Lucas Kanade function used for registration
    * threshold - Threshold for creating binary mask
    * iters - Iterations for morphological closing

    [output]
    * mask: Binary image of input size corresponding to moving objects
    '''

    # Compute affine transform
    M1to2 = np.flip(LK_method(image1, image2))[..., [1, 2, 0]]
    M2to1 = np.linalg.inv(np.vstack([M1to2, [0, 0, 1]]))
    M2to1 = M2to1/M2to1[-1, -1]

    # Warp image
    warp_img = affine_transform(image1, M2to1)
    aff_mask = affine_transform(np.ones(image1.shape), M2to1)
    
    # Get difference image
    diff = np.abs((aff_mask * image2) - (aff_mask * warp_img))
    
    # Create a binary mask using the theshold
    mask = np.where(diff > threshold, 1, 0).astype('bool')
    mask = binary_opening(binary_closing(mask, iterations=iters))

    return mask

if __name__ == '__main__':
    aerialseq = np.load('../data/aerialseq.npy')
    SubtractDominantMotion(aerialseq[:, :, 0], aerialseq[:, :, 1])
