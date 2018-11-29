#!/usr/bin/env python

"""
Display a grid of thumbnails.
"""

import os
import PIL
import skimage.io
import numpy as np
from natsort import natsorted, ns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def display_grid(im_list, grid_shape, axes_pad=0, pad_image=False, cmap='gray'):
    # Grid must be 2D
    assert len(grid_shape) == 2

    # Make sure all images can fit in grid
    assert np.prod(grid_shape) >= len(im_list)

    grid = ImageGrid(plt.gcf(), 111, grid_shape, axes_pad=axes_pad)
    for i in range(len(im_list)):
        data_orig = im_list[i]

        # Scale image
        im = PIL.Image.fromarray(data_orig)
        data = np.array(im)
        if pad_image: data = np.pad(data, ((3, 3), (3, 3)), 'constant', constant_values=(1, 1))
        grid[i].imshow(data, cmap=cmap)

        # Turn off axes
        grid[i].axes.get_xaxis().set_visible(False)
        grid[i].axes.get_yaxis().set_visible(False)
    
    # Display grid
    plt.show()
        
if __name__ == '__main__':
    file_list = natsorted(os.listdir('../crops'), alg=ns.IGNORECASE)
    im_list = [ skimage.img_as_float(skimage.io.imread(os.path.join('../crops', img))) for img in file_list ]
        
    # 233 * 7 = 18 * 7 = 126
    display_grid(im_list[:18*7], (7, 18), pad_image=True)

    # 35 * 222 = 15 * 8 = 120
    display_grid(im_list[18*7:], (8, 15), pad_image=True)
