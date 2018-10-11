#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 2 - Feature Descriptors, Homographies & RANSAC
'''

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from keypointDetect import DoGdetector

TEST_PATTERN_FILE = '../results/testPattern.npy'

def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF
    Run this routine for the given parameters patch_width = 9 and n = 256

    [input]
    * patch_width - Width of the image patch (usually 9)
    * nbits - Number of tests n in the BRIEF descriptor

    [output]
    * compareX and compareY - Linear indices into the patch_width x patch_width
      image patch and are each (nbits,) vectors. 
    '''
    
    # Random
    # compareX = np.random.randint(patch_width, size=(nbits))
    # compareY = np.random.randint(patch_width, size=(nbits))

    # Generation random gaussian distribution spatially
    mean, std = patch_width//2, patch_width/5
    linear_indices = np.arange(0, patch_width*patch_width).reshape(patch_width, patch_width)
    X_gauss_indices = np.clip(np.random.normal(mean, std, (nbits, 2)).round().astype('int'), 0, patch_width-1)
    Y_gauss_indices = np.clip(np.random.normal(mean, std, (nbits, 2)).round().astype('int'), 0, patch_width-1)

    # Linearize the indices
    compareX = np.asarray([ linear_indices[xy[0], xy[1]] for xy in X_gauss_indices ])
    compareY = np.asarray([ linear_indices[xy[0], xy[1]] for xy in Y_gauss_indices ])

    return compareX, compareY

def computeBrief(im, gaussian_pyramid, locsDoG, k, levels, compareX, compareY):
    '''
    Compute BRIEF features
    
    [input]
    * locsDoG - locsDoG are the keypoint locations returned by the DoG detector
    * levels  - Gaussian scale levels that were given in Section 1
    * compareX and compareY - Linear indices into the (patch_width x patch_width)
      image patch and are each (nbits,) vectors

    [output]
    * locs - An m x 3 vector, where the first two columns are the image coordinates
      of keypoints and the third column is the pyramid level of the keypoints
    * desc - An m x n bits matrix of stacked BRIEF descriptors. m is the number of
      valid descriptors in the image and will vary
    '''

    patch_width = 9
    locs, desc = [], []
    H, W, C = gaussian_pyramid.shape

    # (fx, fy) are interest points at a given level
    for fx, fy, level in locsDoG:
        # Check if it's possible to create a patch
        if not ((fy-patch_width//2 >= 0) and (fy+1+patch_width//2 < H) \
            and (fx-patch_width//2 >= 0) and (fx+1+patch_width//2 < W)): continue
        
        # Get a patch of size (patch_width x patch_width)
        patch = gaussian_pyramid[fy-patch_width//2:fy+1+patch_width//2, fx-patch_width//2:fx+1+patch_width//2, level].flatten()
        
        # tx, ty are test points for the patch
        desc.append([ 1 if patch[tx] < patch[ty] else 0 for tx, ty in zip(compareX, compareY) ])
        locs.append([fx, fy, level])
    
    locs = np.asarray(locs)
    desc = np.asarray(desc)
    return locs, desc

def briefLite(im):
    '''
    [input]
    * im - Gray image with values between 0 and 1

    [output]
    * locs - An m x 3 vector, where the first two columns are the image coordinates
      of keypoints and the third column is the pyramid level of the keypoints
    * desc - An m x n bits matrix of stacked BRIEF descriptors
        - m is the number of valid descriptors in the image and will vary
        - n is the number of bits for the BRIEF descriptor
    '''

    locsDoG, gaussian_pyramid = DoGdetector(im)
    locs, desc = computeBrief(im, gaussian_pyramid, locsDoG, np.sqrt(2), 
        [-1, 0, 1, 2, 3, 4], compareX, compareY)

    return locs, desc

def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    outputs : matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches

def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x, y, 'r')
        plt.plot(x, y, 'g.')
    plt.show()

# Test pattern for BRIEF
if os.path.isfile(TEST_PATTERN_FILE):
    print('Loading test pattern from file...')
    compareX, compareY = np.load(TEST_PATTERN_FILE)
else:
    print('Generating new test pattern...')
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'): os.mkdir('../results')
    np.save(TEST_PATTERN_FILE, [compareX, compareY])
    
if __name__ == '__main__':
    # compareX, compareY = makeTestPattern()

    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    # locs, desc = briefLite(im1)  
    
    # fig = plt.figure()
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    # plt.plot(locs[:,0], locs[:,1], 'r.')
    # plt.draw()
    # plt.waitforbuttonpress(0)
    # plt.close(fig)
    
    # Test matches
    # im1 = cv2.imread('../data/pf_desk.jpg')
    # im2 = cv2.imread('../data/pf_scan_scaled.jpg')
    
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1, im2, matches, locs1, locs2)
