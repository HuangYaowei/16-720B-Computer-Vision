#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 2 - Feature Descriptors, Homographies & RANSAC
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

from planarH import computeH

W = np.matrix([[0.0,    18.2,   18.2,   0.0],
               [0.0,    0.0,    26.0,   26.0],
               [1,      1,      1,      1]])

X = np.matrix([[483,    1704,   2175,   67],
               [810,    781,    2217,   2286],
               [1,      1,      1,      1]])

K = np.matrix([[3043.72,    0.0,        1196.00],
               [0.0,        3043.72,    1604.00],
               [0.0,        0.0,        1.0]])

ball_diameter = 6.8581

def load_sphere(filename):
    sphere = np.loadtxt(filename)
    return sphere

def compute_extrinsics(K, H):
    H_new = np.matmul(np.linalg.inv(K), H)
    U, S, VT = np.linalg.svd(H_new[:, :2])

    # Compute rotation
    R12 = np.matmul(np.matmul(U, np.vstack((np.eye(2), [0]*2))), VT)
    R3 = np.cross(R12[:, 0].T, R12[:, 1].T)
    R = np.hstack([R12, R3.T])
    if np.linalg.det(R) < 0: R = np.hstack((R12, R3.T*-1))

    # Compute translation
    scale = np.sum(H_new[:, :2]/R[:, :2])/6
    t = H_new[:, -1]/scale

    return R, t

def project_extrinsics(K, W, R, t):
    proj = np.matmul(R, W + np.matrix([6.16, 18.4, -ball_diameter/2]).T) + t
    proj = np.matmul(K, proj)
    proj = proj/proj[-1]

    return proj

def project_on_image(img, proj):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(proj[0], proj[1], '.', color='yellow')
    plt.show()

if __name__ == '__main__':
    # Load data
    img = cv2.imread('../data/prince_book.jpeg')
    sphere = load_sphere('../data/sphere.txt')

    # Compute parameters
    H = computeH(X[:2], W[:2])
    R, t = compute_extrinsics(K, H)
    
    # Project 3D object on to the image
    proj = project_extrinsics(K, sphere, R, t)
    project_on_image(img, proj)
