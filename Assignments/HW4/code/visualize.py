import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import helper
import submission as sub
from findM2 import findM2

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter3
'''
def visualize():
    # Load data points
    points = np.load('../data/templeCoords.npz')
    
    # Load images
    im1 = scipy.ndimage.imread('../data/im1.png')
    im2 = scipy.ndimage.imread('../data/im2.png')
    
    # Get epipolar data
    F, C1, C2 = findM2()

    # Estimate stereo correspondences
    points1, points2 = [], []
    for i in range(points['x1'].shape[0]):
        points1.append((int(points['x1'][i]), int(points['y1'][i])))
        points2.append(sub.epipolarCorrespondence(im1, im2, F, points1[i][0], points1[i][1]))
    points1, points2 = np.asarray(points1), np.asarray(points2)

    # Triangulate points
    points3D, _ = sub.triangulate(C1, points1, C2, points2)

    # Plot the 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2])
    plt.show()

if __name__ == '__main__':
    visualize()
