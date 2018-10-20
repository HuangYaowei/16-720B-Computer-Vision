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

import matplotlib.pyplot as plt

FAST_WARP = True

def disp(img):
    fig, ax = plt.subplots(1)
    plt.imshow(img, cmap='gray')
    # plt.show()
    plt.draw()
    plt.waitforbuttonpress(0) # this will wait for indefinite time
    plt.close(fig)

def crop(img, rect):
    # Slower accurate method
    if not FAST_WARP:
        x_range = np.linspace(0, img.shape[1], img.shape[1], endpoint=False)
        y_range = np.linspace(0, img.shape[0], img.shape[0], endpoint=False)
        rect_x = np.linspace(rect[0], rect[2], rect[2]-rect[0], endpoint=False)
        rect_y = np.linspace(rect[1], rect[3], rect[3]-rect[1], endpoint=False)
        warped = RectBivariateSpline(y_range, x_range, img)(rect_y, rect_x)
    
    # Faster approx method
    else:
        rect = rect.round().astype('int')
        warped = img[rect[1]:rect[3], rect[0]:rect[2]]

    return warped

def LucasKanade(It, It1, rect, p0=np.zeros(2), threshold=0.001, iters=20, debug=None):
    '''
    [input]
    * It - Template image
    * It1 - Current image
    * rect - Current position of the car (top left, bot right coordinates)
    * p0 - Initial movement vector [dp_x0, dp_y0]
    * threshold - Threshold for error convergence (default: 0.001)
    * iters - Number of iterations for error convergence (default: 20)
    
    [output]
    * p - Movement vector [dp_x, dp_y]
    '''

    # Initial parameters
    p = p0

    # Iterate 
    for i in range(iters):
        # Step 1 - Warp image
        warp_img = shift(It1, np.flip(-p))

        # Step 2 - Compute error image
        error_img = It - crop(warp_img, rect)

        # Step 3 - Warp the gradient
        gradient = np.dstack([np.gradient(warp_img, axis=1), np.gradient(warp_img, axis=0)])
        gradient = np.dstack([crop(gradient[:, :, 0], rect), crop(gradient[:, :, 1], rect)])
        warp_gradient = gradient.reshape(gradient.shape[0] * gradient.shape[1], 2)

        # Step 4 - Evaluate jacobian
        jacobian = np.eye(2)

        # Step 5 - Compute the steepest descent images
        steepest_descent = np.matmul(warp_gradient, jacobian)

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
    carseq = np.load('../data/carseq.npy')
    LucasKanade(carseq[:, :, 0], carseq[:, :, 1], np.asarray([59, 116, 145, 151]))
