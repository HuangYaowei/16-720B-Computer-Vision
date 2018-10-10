import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from planarH import computeH
from keypointDetect import displayImage, displayPoints

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

def plot3D(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[0], points[1], points[2])
    plt.grid()
    plt.axis('equal')
    plt.show()

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
    H44 = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]])
    proj = np.matmul(K, np.matmul(H44, W)[:-1])
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

    # Translate sphere
    sphere[0] += 10.65
    sphere[1] += 20.45
    sphere[2] -= ball_diameter/2

    # Compute parameters
    H = computeH(X[:2], W[:2])
    R, t = compute_extrinsics(K, H)
    W = np.vstack([sphere, [1]*sphere.shape[1]])
    
    # Project 3D object on to the image
    proj = project_extrinsics(K, W, R, t)
    project_on_image(img, proj)
