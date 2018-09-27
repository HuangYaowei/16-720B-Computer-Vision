import numpy as np
import cv2

# TODO: Remove
import sys

def createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4]):
    # Convert RGB to grayscale
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Convert (0-255) to (0-1) range
    if im.max() > 10:
        im = np.float32(im)/255

    # Create image pyramid
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0, 0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, 
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1, 0, 1, 2, 3, 4]):
    '''
    Produces DoG Pyramid
    
    [input]
    * Gaussian Pyramid - A matrix of grayscale images of size [imH, imW, len(levels)]
    * levels - The levels of the pyramid where the blur at each level is outputs
    
    [output]
    * DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid 
    created by differencing the Gaussian Pyramid input
    * DoG_levels - The levels of the DoG pyramid 
    '''

    # Create DoG pyramid
    DoG_pyramid = []
    for i in range len(levels - 1):
        DoG_pyramid.append(gaussian_pyramid[:, :, i+1] - gaussian_pyramid[:, :, i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)    

    # DoG levels
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    [input]
    * DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    [output]
    * principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
    point contains the curvature ratio R for the corresponding point in the DoG pyramid
    '''
    
    principal_curvature = None
 

    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    [input]
    * DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    * DoG_levels - The levels of the pyramid where the blur at each level is outputs
    * principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the curvature ratio R
    * th_contrast - remove any point that is a local extremum but does not have a
    DoG response magnitude above this threshold
    * th_r - remove any edge-like points that have too large a principal curvature ratio
     
    [output]
    * locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
    scale and space, and also satisfies the two thresholds
    '''
    
    locsDoG = None
   


    return locsDoG
    
def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4], th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    
    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''

    return locsDoG, gauss_pyramid

if __name__ == '__main__':
    levels = [-1, 0, 1, 2, 3, 4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    
    # Test gaussian pyramid
    im_pyr = createGaussianPyramid(im)
    # displayPyramid(im_pyr)

    # Test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    print(DoG_pyr.shape, DoG_levels)
    displayPyramid(DoG_pyr)

    sys.exit(0)
    
    # Test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    displayPyramid(pc_curvature)
    
    # Test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    
    # Test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
