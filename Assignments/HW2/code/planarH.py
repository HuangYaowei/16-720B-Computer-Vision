import numpy as np
import cv2

from BRIEF import briefLite, briefMatch
from keypointDetect import displayPoints

def computeH(p1, p2):
    '''
    [input]
    * p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)
      coordinates between two images

    [output]
    * H2to1 - A 3 x 3 matrix encoding the homography that best 
      matches the linear equation
    '''

    assert(p1.shape[1] == p2.shape[1])
    assert(p1.shape[0] == 2)

    A = []
    for i in range(p1.shape[1]):
        A.append([-p1[0, i], -p1[1, i], -1, 0, 0, 0, p1[0, i]*p2[0, i], p2[0, i]*p1[1, i], p2[0, i]])
        A.append([0, 0, 0, -p1[0, i], -p1[1, i], -1, p1[0, i]*p2[1, i], p2[1, i]*p1[1, i], p2[1, i]])

    U, S, V = np.linalg.svd(np.asarray(A))
    H2to1 = V.T[:, -1].reshape(3, 3)/V.T[-1, -1]

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using RANSAC

    [input]
    * locs1 and locs2 - Matrices specifying point locations in each of the images
    * matches - Matrix specifying matches between these two sets of point locations
    * nIter - Number of iterations to run RANSAC
    * tol - Tolerance value for considering a point to be an inlier

    [output]
    * bestH - Homography matrix with the most inliers found during RANSAC
    ''' 

    p1, p2 = [], []
    for i in range(matches.shape[0]):
        p1.append(locs1[matches[i, 0], 0:2])
        p2.append(locs2[matches[i, 1], 0:2])

    p1 = np.asarray(p1).swapaxes(0, 1)
    p2 = np.asarray(p2).swapaxes(0, 1)

    ng = 4
    np.random.seed(0)
    groups = np.random.permutation(np.arange(0, p1.shape[1]))
    if p1.shape[1]%ng: groups = np.concatenate(groups, np.random.randint(p1.shape[1], size=p1.shape[1]%ng))

    for i in range(groups.shape[0]//ng):
        H = computeH(p1[:, i:i+ng], p2[:, i:i+ng])

    a = np.vstack((p1[:, :4], [1]*4))
    b = np.vstack((p2[:, :4], [1]*4))
    
    print(a, a.shape)
    print(b, b.shape)
    print(H)
    
    dst = np.matmul(H, a, dst)
    print(dst/dst[-1, :])

    bestH = None
    return bestH

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')

    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    
    locs1, locs2, desc1, desc2, matches = np.load('BRIEF-ch.npy')
    ret = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    
    # print(len(matches))
    # displayPoints(im2, ret)
