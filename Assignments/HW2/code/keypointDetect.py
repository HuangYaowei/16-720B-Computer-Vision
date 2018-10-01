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
    while (cv2.waitKey(33) != 27): pass
    cv2.destroyAllWindows()

def displayImage(img, title='Image'):
    # Single image
    if type(img) == np.ndarray:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, img)
        cv2.resizeWindow(title, 500, 500)

    # Multiple images
    else: 
        for i, im in enumerate(img): cv2.imshow(title + str(i), im)

    while (cv2.waitKey(33) != 27): pass
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
    for i in range(len(levels) - 1):
        DoG_pyramid.append(gaussian_pyramid[:, :, i+1] - gaussian_pyramid[:, :, i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)    

    # DoG levels
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoG Pyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    [input]
    * DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    [output]
    * principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
      point contains the curvature ratio R for the corresponding point in the DoG pyramid
    '''

    height, width, channels = DoG_pyramid.shape
    principal_curvature = []

    for i in range(channels):
        img = DoG_pyramid[:, :, i]

        # First derivatives
        dx = cv2.Sobel(img, ddepth=-1, dx=1, dy=0)
        dy = cv2.Sobel(img, ddepth=-1, dx=0, dy=1)

        # Second derivatives
        dxx = cv2.Sobel(dx, ddepth=-1, dx=1, dy=0)
        dyy = cv2.Sobel(dy, ddepth=-1, dx=0, dy=1)
        dxy = cv2.Sobel(dx, ddepth=-1, dx=0, dy=1)

        R = np.zeros((height, width))
        for r in range(height):
            for c in range(width):
                H = np.asarray([[dxx[r, c], dxy[r, c]], [dxy[r, c], dyy[r, c]]])
                R[r, c] = (np.trace(H)**2)/np.linalg.det(H)
        principal_curvature.append(R)

    principal_curvature = np.dstack(principal_curvature)
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoG Pyramid

    [input]
    * DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    * DoG_levels - The levels of the pyramid where the blur at each level is outputs
    * principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the curvature ratio R
    * th_contrast - Remove any point that is a local extremum but does not have a
      DoG response magnitude above this threshold
    * th_r - Remove any edge-like points that have too large a principal curvature ratio
     
    [output]
    * locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
      scale and space, and also satisfies the two thresholds
    '''
    
    H, W, C = DoG_pyramid.shape
    locsDoG = []
    
    for level in range(0, C):
        for x in range(0, H):
            for y in range(0, W):
                # Check principal curvature ratio
                if abs(principal_curvature[x, y, level]) > th_r: continue
                
                # Get scale-space neighbours
                neighbours = DoG_pyramid[x-1*int(x>0):x+2*int(x<H), y-1*int(y>0):y+2*int(y<W), level].flatten()
                neighbours = np.concatenate([neighbours, DoG_pyramid[x, y, level-1*int(level>0):level+2**int(level<C)]])
                
                # Check if current pixel is an extrema
                if (np.max(neighbours) != DoG_pyramid[x, y, level]) \
                and (np.min(neighbours) != DoG_pyramid[x, y, level]): continue

                # Check contrast ratio
                if (not abs(DoG_pyramid[x, y, level]) > th_contrast): continue

                # Add this point of interest
                locsDoG.append([y, x, level])

    locsDoG = np.asarray(locsDoG)
    return locsDoG

def displayPoints(img, locsDoG, factor=10, fname=None):
    # Resize for better display
    img = cv2.resize(img, None, fx=factor, fy=factor)
    
    # Draw the interest points
    for x, y, level in locsDoG:
        color = (int(level==1)*255, int(level==2)*255, int(level==3)*255)
        img = cv2.circle(img, (x*factor, y*factor), factor, (0, 255, 0), -1)
    
    if fname: cv2.imwrite(fname, img)
    displayImage(img)

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4], th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    [input]    
    * im - Grayscale image with range [0, 1]
    * sigma0 - Scale of the 0th image pyramid
    * k - Pyramid factor (sqrt(2))
    * levels - Levels of pyramid to construct (-1:4)
    * th_contrast - DoG contrast threshold (0.03)
    * th_r - Principal Ratio threshold (12)

    [output]
    * locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema 
      in both scale and space, and satisfies the two thresholds
    * gauss_pyramid - A matrix of grayscale images of size (H, W, len(levels))
    '''

    # Get gaussian pyramid
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)

    # Get DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    
    # Compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    
    # Get interest points
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    # Display the points
    # displayPoints(im, locsDoG)

    return locsDoG, gauss_pyramid

if __name__ == '__main__':
    levels = [-1, 0, 1, 2, 3, 4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    
    '''
    # Test gaussian pyramid
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)

    # Test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    
    # Test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    displayPyramid(pc_curvature)
    
    # Test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    # Display the points
    displayPoints(im, locsDoG)
    '''
    
    # Test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
