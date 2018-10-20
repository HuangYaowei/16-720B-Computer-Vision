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

from LucasKanade import LucasKanade, crop

def play(filename):
    frames = np.load(filename)
    total_frames = frames.shape[2]
    fps = 24

    # Init figure
    fig, ax = plt.subplots(1)
    im = ax.imshow(frames[:, :, 0], animated=True, cmap='gray')

    # Initial rect
    init_rect = np.asarray([59, 116, 145, 151])
    rect = init_rect

    def updatefig(j):
        nonlocal rect
        sys.stdout.write('\rFrame: %04d/%04d | FPS: %d | Sequence: [%s]%s' % (j, total_frames, fps, filename, ' '*10))

        # First frame rect
        if j == 1: rect = init_rect

        # Update image for display
        im.set_array(frames[:, :, j])

        # Lucas-Kanade using latest template
        template = crop(frames[:, :, j-1], rect)
        p = LucasKanade(template, frames[:, :, j], rect)
        rect = (rect.reshape(2, 2) + p).flatten()

        # Clear patches and draw new boxes
        for patch in reversed(ax.patches): patch.remove()
        box = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], linewidth=2, edgecolor='y', facecolor='none')
        ax.add_patch(box)

        return im

    # Run animation and display window
    ani = animation.FuncAnimation(fig, updatefig, frames=range(1, total_frames), interval=1000//fps)
    plt.show()

if __name__ == '__main__':
    play('../data/carseq.npy')
