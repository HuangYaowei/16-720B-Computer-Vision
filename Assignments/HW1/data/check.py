import os
import cv2
import numpy as np
from scipy import ndimage

folders = ['kitchen', 'baseball_field', 'waterfall', 'highway', 
           'desert', 'windmill', 'laundromat', 'auditorium']

def get_random_window_slice(shape, alpha):
    '''
    TODO: Document
    NOTE: Max alpha value for the dataset possible is 186
    '''

    if alpha >= min(shape[:1]):
        return slice(min(shape[:1])), slice(min(shape[:1]))

    ran_h = np.random.choice(shape[0] - alpha)
    ran_w = np.random.choice(shape[1] - alpha)
    
    h_slice = slice(ran_h, ran_h + alpha)
    w_slice = slice(ran_w, ran_w + alpha)

    print(shape, ran_h, ran_h + alpha, ran_w, ran_w + alpha)
    
    return h_slice, w_slice

if __name__ == '__main__':
    for folder in folders:
        files = os.listdir(folder)

        for f in files:
            try:
                img = ndimage.imread('%s/%s'%(folder, f))
                # print(img.shape, f)
            except Exception as e:
                print(f, e)
