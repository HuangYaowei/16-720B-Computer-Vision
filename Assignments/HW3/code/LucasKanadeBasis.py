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
from scipy.ndimage import shift
from scipy.interpolate import RectBivariateSpline

from LucasKanade import crop, disp
    
def LucasKanadeBasis(It, It1, rect, bases, p0=np.zeros(2), threshold=0.001, iters=20):
    '''
    [input]
    * It - Template image
    * It1 - Current image
    * rect - Current position of the car (top left, bot right coordinates)
    * bases - [n, m, k] where (n x m) is the size of the template
    * threshold - Threshold for error convergence (default: 0.001)
    * iters - Number of iterations for error convergence (default: 20)
    
    [output]
    * p - Movement vector [dp_x, dp_y]
    '''

    # Initial parameters
    p = p0
    bases = bases.reshape(bases.shape[0] * bases.shape[1], bases.shape[2])

    # Iterate 
    for i in range(iters):
        # Step 1 - Warp image
        warp_img = shift(It1, np.flip(-p))

        # Step 2 - Compute error image
        error_img = It - crop(warp_img, rect)
        # error_img = error_img.flatten() - np.matmul(bases, np.matmul(bases.T, error_img.flatten()))

        # Step 3 - Compute and warp the gradient
        gradient = np.dstack(np.gradient(warp_img)[::-1])
        gradient = np.dstack([crop(gradient[:, :, 0], rect), crop(gradient[:, :, 1], rect)])
        warp_gradient = gradient.reshape(gradient.shape[0] * gradient.shape[1], 2)

        # Step 4 - Evaluate jacobian
        jacobian = np.eye(2)

        # Step 5 - Compute the steepest descent images
        steepest_descent = np.matmul(warp_gradient, jacobian)
        steepest_descent = steepest_descent - np.matmul(bases, np.matmul(bases.T, steepest_descent))
        
        # Step 6 - Compute the Hessian matrix
        hessian = np.matmul(steepest_descent.T, steepest_descent)

        # Step 7/8 - Compute delta P
        delta_p = np.matmul(np.linalg.inv(hessian), np.matmul(steepest_descent.T, error_img.flatten()))
        
        # Step 9 - Update the parameters
        p = p + delta_p

        # Test for convergence
        if np.linalg.norm(delta_p) <= threshold: break

    return p

if __name__ == '__main__':
    frames = np.load('../data/sylvseq.npy')    
    bases = np.load('../data/sylvbases.npy')
    rect = np.asarray([101, 61, 155, 107])
    LucasKanadeBasis(crop(frames[:, :, 0], rect), frames[:, :, 1], rect, bases)
