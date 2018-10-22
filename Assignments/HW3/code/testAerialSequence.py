#!/usr/bin/python3

'''
16-720B Computer Vision (Fall 2018)
Homework 3 - Lucas-Kanade Tracking and Correlation Filters
'''

__author__ = "Heethesh Vhavle"
__credits__ = ["Simon Lucey", "16-720B TAs"]
__version__ = "1.0.1"
__email__ = "heethesh@cmu.edu"

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from SubtractDominantMotion import SubtractDominantMotion
from InverseCompositionAffine import InverseCompositionAffine

def play(filename):
    frames = np.load(filename)
    total_frames = frames.shape[2]
    fps = 24

    # Init figure
    fig, ax = plt.subplots(1)
    im = ax.imshow(frames[:, :, 0], animated=True, cmap='gray')

    def updatefig(j):
        sys.stdout.write('\rFrame: %04d/%04d | FPS: %d | Sequence: [%s]%s' % (j, total_frames, fps, filename, ' '*10))

        # Update image for display
        ax.clear()
        im = ax.imshow(frames[:, :, j], cmap='gray')

        # Get motion detection mask and display
        mask = SubtractDominantMotion(frames[:, :, j-1], frames[:, :, j])#, InverseCompositionAffine)
        ax.imshow(np.ma.masked_array(mask, np.invert(mask)), alpha=0.7, cmap='autumn')

        return im

    # Run animation and display window
    ani = animation.FuncAnimation(fig, updatefig, frames=range(1, total_frames), interval=1000//fps)
    plt.show()

if __name__ == '__main__':
    play('../data/aerialseq.npy')