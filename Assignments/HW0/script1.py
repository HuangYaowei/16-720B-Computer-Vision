# Problem 1: Image Alignment
import sys
import cv2
import time
import numpy as np 

X_OFFSET_RANGE = 30
Y_OFFSET_RANGE = 30

OPTIMISE = True
OPTIMISE_STRIDE = 3
OPTIMISE_ROI_SIZE = 10
OPTMISE_MAGIC_REGION = 475

def display(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, int(image.shape[0]/2), int(image.shape[1]/2))
    cv2.moveWindow(name, 50, 50)
    cv2.imshow(name, image)

def size(image):
    print ('Image Size:', image.shape)

def main():
    # Load images (all 3 channels)
    red = np.load('red.npy')
    green = np.load('green.npy')
    blue = np.load('blue.npy')

    start = time.time()

    half_h = OPTMISE_MAGIC_REGION if OPTIMISE else int(red.shape[0]/2)
    half_w = OPTMISE_MAGIC_REGION if OPTIMISE else int(red.shape[1]/2)

    if OPTIMISE: roi_offset = [OPTIMISE_ROI_SIZE, OPTIMISE_ROI_SIZE]
    else: roi_offset = [half_h, half_w]
    
    roi_h = (half_h - roi_offset[0], half_h + roi_offset[0])
    roi_w = (half_w - roi_offset[1], half_w + roi_offset[1])

    stride = OPTIMISE_STRIDE if OPTIMISE else 1

    ssd_min_rg = sys.maxsize
    ssd_min_rb = sys.maxsize

    ssd_shift_rg = None
    ssd_shift_rb = None

    for xshift in range(-X_OFFSET_RANGE, X_OFFSET_RANGE + 1, stride):
        for yshift in range(-Y_OFFSET_RANGE, Y_OFFSET_RANGE + 1, stride):
            red_shifted = np.roll(red, [yshift, xshift], axis=(0,1))
            red_roi = red_shifted[roi_h[0]:roi_h[1], roi_w[0]:roi_w[1]]

            ssd_rg = np.sum((red_roi - green[roi_h[0]:roi_h[1], roi_w[0]:roi_w[1]]) ** 2)
            if ssd_rg < ssd_min_rg:
                ssd_min_rg = ssd_rg
                ssd_shift_rg = [-yshift, -xshift]
            
            ssd_rb = np.sum((red_roi - blue[roi_h[0]:roi_h[1], roi_w[0]:roi_w[1]]) ** 2)
            if ssd_rb < ssd_min_rb:
                ssd_min_rb = ssd_rb
                ssd_shift_rb = [-yshift, -xshift]

    print('Red-Green:')
    print(ssd_min_rg, ssd_shift_rg)

    print('\nRed-Blue:')
    print(ssd_min_rb, ssd_shift_rb)

    # Perform the shift
    green_shift = np.roll(green, ssd_shift_rg, axis=(0, 1))
    blue_shift = np.roll(blue, ssd_shift_rb, axis=(0, 1))

    print('Time taken:', time.time() - start)

    display('Raw Merged', cv2.merge((blue, green, red)))
    display('Merged', cv2.merge((blue_shift, green_shift, red)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
