import cv2
import numpy as np

from planarH import computeH
from keypointDetect import displayImage, displayPoints

W = np.matrix([[0.0, 18.2, 18.2, 0.0],
               [0.0, 0.0, 26.0, 26.0],
               [0.0, 0.0, 0.0, 0.0]])

X = np.matrix([[483, 1704, 2175, 67],
               [810, 781, 2217, 2286],
               [1, 1, 1, 1]])

K = np.matrix([[3043.72, 0.0, 1196.00],
               [0.0, 3043.72, 1604.00],
               [0.0, 0.0, 1.0]])

if __name__ == '__main__':
    img = cv2.imread('../data/prince_book.jpeg')
    print(img.shape)


    H = computeH(X[:2], W[:2])
    print(H)
    print(np.matmul(K, np.matmul(H, X)))

    # displayImage(img, 'Input', 650)
    # displayPoints(img, X.T.tolist(), factor=1, radius=40)