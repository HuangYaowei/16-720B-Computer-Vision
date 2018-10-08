import sys

import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from planarH import ransacH
from keypointDetect import displayImage
from BRIEF import briefLite, briefMatch, plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''

    H1to2 = np.linalg.inv(H2to1)
    H1to2 = H1to2/H1to2[-1, -1]
    
    top_left = np.matmul(H1to2, np.asarray([0, 0, 1]))
    bottom_right = np.matmul(H1to2, np.asarray([im2.shape[1], im2.shape[0], 1]))
    bottom_right = bottom_right/bottom_right[-1]
    
    # offsetx = abs(int(top_left[0]))
    # offsety = abs(int(top_left[1]))
    # dsize = (int(bottom_right[0]) + offsetx, int(bottom_right[1]) + offsety)

    print(top_left, bottom_right)
    pano_im = cv2.warpPerspective(im2, H1to2, dsize=(2000, 1000))
    pano_im[0:im1.shape[0], 0:im1.shape[1]] = im1


     # = cv2.warpPerspective(im1, H2to1, dsize=(1000, 1500))

    # pano_im = cv2.warpPerspective(im2, H2to1, (im2.shape[1] + im1.shape[1], im2.shape[0]))
    # pano_im[0:im1.shape[0], 0:im1.shape[1]] = im1

    return pano_im

def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 

    return pano_im

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    
    print(im1.shape)
    
    # print('Computing BRIEF features for im1...')
    # locs1, desc1 = briefLite(im1)
    # print('Computing BRIEF features for im2...')
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    
    # np.save('incline', [locs1, locs2, desc1, desc2, matches])
    locs1, locs2, desc1, desc2, matches = np.load('incline.npy')

    # plotMatches(im1,im2,matches,locs1,locs2)
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # np.save('H2to1', H2to1)
    H2to1 = np.load('H2to1.npy')
    print(H2to1)

    # pano_im = imageStitching_noClip(im1, im2, H2to1)
    pano_im = imageStitching(im1, im2, H2to1)

    # cv2.imwrite('../results/panoImg.png', pano_im)
    displayImage(pano_im,'panoramas')