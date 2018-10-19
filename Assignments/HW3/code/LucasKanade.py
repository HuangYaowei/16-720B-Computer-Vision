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

def disp(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def warp(img, rect):
    H, W = img.shape
    y_range, x_range = np.linspace(0, H, H, endpoint=False), np.linspace(0, W, W, endpoint=False)
    rect_y, rect_x = np.linspace(rect[1], rect[3], rect[3]-rect[1], endpoint=False), np.linspace(rect[0], rect[2], rect[2]-rect[0], endpoint=False)
    warped = RectBivariateSpline(y_range, x_range, img)(rect_y, rect_x)
        return warped

def LucasKanade(It, It1, rect, p0=np.zeros(2), threshold=0.001, iters=20):
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

    p = p0
    template = warp(It, rect)

    for i in range(iters):
        warp_img = shift(It1, -p)

        gradient = np.dstack([np.gradient(warp_img, axis=1), np.gradient(warp_img, axis=0)])
        gradient = np.dstack([warp(gradient[:, :, 0], rect), warp(gradient[:, :, 1], rect)])
        warp_gradient = gradient.reshape(gradient.shape[0] * gradient.shape[1], 2)

        jacobian = np.eye(2)
        steepest_descent = np.matmul(warp_gradient, jacobian)
        hessian = np.matmul(steepest_descent.T, steepest_descent)

        error_img = template - warp(warp_img, rect)
        delta_p = np.matmul(np.linalg.inv(hessian), np.matmul(steepest_descent.T, error_img.flatten()))
        
        p = p + np.flip(delta_p)
        if np.linalg.norm(delta_p) <= threshold: break

    return np.flip(p)

if __name__ == '__main__':
    carseq = np.load('../data/carseq.npy')
    LucasKanade(carseq[:, :, 0], carseq[:, :, 1], np.asarray([59, 116, 145, 151]))