import sys

import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

from planarH import ransacH
from keypointDetect import displayImage
from BRIEF import briefLite, briefMatch, plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix

    [input]
    * Warps img2 into img1 reference frame using the provided warpH() function
    * H2to1 - A 3 x 3 matrix encoding the homography that best matches the linear equation

    [output]
    * pano_im - Blends img1 and warped img2 and outputs the panorama image
    '''
    
    # Arbitrary output size
    output_size = (1750, 700)

    # Warp the image and combine the panorama without clipping correction
    im1_warp = cv2.warpPerspective(im1, np.eye(3), dsize=output_size)
    im2_warp = cv2.warpPerspective(im2, H2to1, dsize=output_size)

    # Blend the images to form the panorama
    pano_im = np.maximum(im1_warp, im2_warp)
    return pano_im

def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix without clipping

    [input]
    * Warps img2 into img1 reference frame using the provided warpH() function
    * H2to1 - A 3 x 3 matrix encoding the homography that best matches the linear equation
    
    [output]
    * pano_im - Blends img1 and warped img2 and outputs the panorama image without clipping
    ''' 

    # Required output width
    output_width = 2000

    # Collect the corner points of the warped image
    corners = np.asarray([[0, 0, 1], [0, im2.shape[0]-1, 1], [im2.shape[1]-1, 0, 1], [im2.shape[1]-1, im2.shape[0]-1, 1]]).T
    corners = np.matmul(H2to1, corners)
    corners = (corners/corners[-1, :]).astype('int').T

    # Find the translation offset and maximum coordinates of the warped image
    minXY = np.min(corners, axis=0)[:2]
    maxXY = np.max(corners, axis=0)[:2]
    offset = np.asarray([ -i if i<0 else 0 for i in minXY ])

    # Compute the output size, aspect ratio and the required scaling
    output_size = (maxXY[0] + offset[0], maxXY[1] + offset[1])
    aspect_ratio = output_size[1]/output_size[0]
    scale = (output_width * aspect_ratio)/output_size[1]

    # Update the offset and output size with the scaling factor
    offset = (offset * scale).astype('int')
    output_size = (int(output_size[0] * scale), int(output_size[1] * scale))

    # Create the panorama by warping with the H and M matrices
    M = np.asarray([[scale, 0, offset[0]], [0, scale, offset[1]], [0, 0, 1]]).astype('float')
    im1_warp = cv2.warpPerspective(im1, M, dsize=output_size)
    im2_warp = cv2.warpPerspective(im2, np.matmul(M, H2to1), dsize=output_size)
    
    # Blend the images to form the panorama
    pano_im = np.maximum(im1_warp, im2_warp)
    return pano_im

def generatePanorama(im1, im2):
    '''
    Accepts two images as input, computes keypoints and descriptors for 
    both the images, finds putative feature correspondences by matching keypoint
    descriptors, estimates a homography using RANSAC and then warps one of the
    images with the homography so that they are aligned and then overlays them

    [input]
    * im1 - Input image 1
    * im2 - Input image 2

    [output]
    * pano_im - Output panorama image
    '''

    # Compute keypoints and descriptors
    print('Computing feature descriptors for im1...')
    locs1, desc1 = briefLite(im1)

    print('Computing feature descriptors for im2...')
    locs2, desc2 = briefLite(im2)
    
    # Match keypoint descriptors
    matches = briefMatch(desc1, desc2)

    # Estimate homography
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

    # Align and blend the images to form the panorama
    pano_im = imageStitching_noClip(im1, im2, H2to1)
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
    # plotMatches(im1, im2, matches, locs1, locs2)

    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # H2to1 = np.load('H2to1_correct.npy')

    # pano_im = imageStitching(im1, im2, H2to1)
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    # cv2.imwrite('../results/panoImg.png', pano_im)
    displayImage(pano_im, 'panoramas')
