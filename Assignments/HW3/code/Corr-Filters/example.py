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
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import correlate, convolve

img = np.load('lena.npy')

# Template cornes in image space [[x1, x2, x3, x4], [y1, y2, y3, y4]]
pts = np.array([[248, 292, 248, 292],
                [252, 252, 280, 280]])

# Size of the template (h, w)
dsize = np.array([pts[1, 3] - pts[1, 0] + 1,
                  pts[0, 1] - pts[0, 0] + 1])

# Set template corners
tmplt_pts = np.array([[0, dsize[1]-1, 0, dsize[1], -1],
                      [0, 0, dsize[0] - 1, dsize[0] - 1]])

# Apply warp p to template region of img
def imwarp(p):
    global img, dsize
    return img[p[1]:(p[1]+dsize[0]), p[0]:(p[0]+dsize[1])]

# Get positive example
gnd_p = np.array([252, 248])  # ground truth warp
x = imwarp(gnd_p)  # the template

# Set up figure
fig, axarr = plt.subplots(1, 3)
axarr[0].imshow(img, cmap='gray')
patch = patches.Rectangle((gnd_p[0], gnd_p[1]), dsize[1], dsize[0], linewidth=1, edgecolor='r', facecolor='none')
axarr[0].add_patch(patch)
axarr[0].set_title('Image')

cropax = axarr[1].imshow(x, cmap='gray')
axarr[1].set_title('Cropped Image')

dx = np.arange(-np.floor(dsize[1]/2), np.floor(dsize[1]/2)+1, dtype=int)
dy = np.arange(-np.floor(dsize[0]/2), np.floor(dsize[0]/2)+1, dtype=int)
[dpx, dpy] = np.meshgrid(dx, dy)
dpx = dpx.reshape(-1, 1)
dpy = dpy.reshape(-1, 1)
dp = np.hstack((dpx, dpy))
N = dpx.size

all_patches = np.ones((N*dsize[0], dsize[1]))
all_patchax = axarr[2].imshow(all_patches, cmap='gray', aspect='auto', norm=colors.NoNorm())
axarr[2].set_title('Concatenation of Sub-Images (X)')

sigma = 5
X = np.zeros((N, N))
Y = np.zeros((N, 1))

def get_filter(X, y, lambda_):
    S = np.matmul(X, X.T)
    temp = (lambda_ * np.eye(X.shape[0]))
    temp = np.linalg.inv(S + temp)

    g = np.matmul(temp, np.matmul(X, y))
    g = g.reshape(dsize)

    print(g.shape)

    return g

def display(img, title):
    fig_disp, ax_disp = plt.subplots(1)
    ax_disp.imshow(img, cmap='gray')
    plt.title(title)
    plt.autoscale(tight=True)
    plt.show()

def display_filter(g, n):
    display(g, 'Filter Visualisation for $\lambda = %d$'%n)

def apply_filter(img, g, n, method=correlate):
    # Flip filter for convolve
    if method is convolve: g = g[::-1, ::-1]
    
    # Apply filter
    output = method(img, g)

    # Display
    display(output, 'Filter Response for $\lambda = %d$ using %s'%(n, method.__name__))

def init():
    return [cropax, patch, all_patchax]

def animate(i):
    global X, Y, dp, gnd_p, sigma, all_patches, patch, cropax, all_patchax, N

    # If the animation is still running
    if i < N:
        xn = imwarp(dp[i, :] + gnd_p)
        X[:, i] = xn.reshape(-1)
        Y[i] = np.exp(-np.dot(dp[i, :], dp[i, :])/sigma)
        all_patches[(i*dsize[0]):((i+1)*dsize[0]), :] = xn
        cropax.set_data(xn)
        all_patchax.set_data(all_patches.copy())
        all_patchax.autoscale()
        patch.set_xy(dp[i, :] + gnd_p)
        return [cropax, patch, all_patchax]

    # Stuff to do after the animation ends
    else:
        fig3d = plt.figure()
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.plot_surface(dpx.reshape(dsize), dpy.reshape(dsize), Y.reshape(dsize), cmap='coolwarm')

        # Place your solution code for question 4.3 here
        g = get_filter(X, Y, lambda_=0)
        display_filter(g, 0)
        apply_filter(img, g, 0, correlate)

        plt.show()
        return []

# Start the animation
ani = animation.FuncAnimation(fig, animate, frames=N+1, init_func=init, blit=True, repeat=False, interval=10)
plt.show()
