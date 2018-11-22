#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 4 - 3D Reconstruction
'''

__author__ = "Heethesh Vhavle"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"
__email__ = "heethesh@cmu.edu"

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import helper
import submission as sub
from findM2 import findM2

def plot3D(points3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])
    plt.show(block=True)

def visualize(pts1, F, C1, C2, im1, im2):
    # Estimate stereo correspondences
    pts2 = np.asarray([ sub.epipolarCorrespondence(im1, im2, F, pts1[i, 0], pts1[i, 1]) for i in range(pts1.shape[0]) ])

    # Triangulate points
    points3D, _ = sub.triangulate(C1, pts1, C2, pts2)

    # Plot the 3D points
    plot3D(points3D)
    
'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter3
'''
if __name__ == '__main__':
    # Load data points
    points = np.load('../data/templeCoords.npz')
    some_corresp = np.load('../data/some_corresp.npz')
    intrinsics = np.load('../data/intrinsics.npz')

    # Load images
    im1 = scipy.ndimage.imread('../data/im1.png')
    im2 = scipy.ndimage.imread('../data/im2.png')

    # Compute fundamental matrix
    F = sub.eightpoint(some_corresp['pts1'], some_corresp['pts2'], 640)
    
    # Get epipolar data
    C1, C2, M1, M2, P = findM2(some_corresp['pts1'], some_corresp['pts2'], F, intrinsics['K1'], intrinsics['K2'])
    pts1 = np.hstack((points['x1'], points['y1'])).astype('int')
    
    # Run correspondence and visualize point cloud
    visualize(pts1, F, C1, C2, im1, im2)
