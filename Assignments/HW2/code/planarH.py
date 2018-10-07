import numpy as np
import cv2

from BRIEF import briefLite, briefMatch, plotMatches
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
    * matches - Matrix specifying matches between these two sets of point locations
    * locs1 and locs2 - Matrices specifying point locations in each of the images
    * num_iter - Number of iterations to run RANSAC
    * tol - Tolerance value for considering a point to be an inlier

    [output]
    * bestH - Homography matrix with the most inliers found during RANSAC
    ''' 

    n_matches = matches.shape[0]

    p1, p2 = [], []
    for i in range(n_matches):
        p1.append(locs1[matches[i, 0], 0:2])
        p2.append(locs2[matches[i, 1], 0:2])

    p1 = np.asarray(p1).swapaxes(0, 1)
    p2 = np.asarray(p2).swapaxes(0, 1)

    # RANSAC
    bestH, most_inliers = None, 0
    best_args = None
    for i in range(num_iter):
        random_indices = np.random.randint(n_matches, size=4)
        H = computeH(p1[:, random_indices], p2[:, random_indices])

        point1 = np.vstack((p1, [1]*n_matches))
        point2 = np.vstack((p2, [1]*n_matches))
        point2_proj = np.matmul(H, point1)
        point2_proj = point2_proj/point2_proj[-1, :]
        
        # distances = np.sqrt(np.sum((point2_proj.T - point2.T)**2, axis=1))
        distances = np.linalg.norm(point2_proj.T - point2.T, axis=1)
        inliers = len(distances[distances<tol])
        
        # TODO: Remove args
        args = [ k for k in range(n_matches) if distances[k] < tol ]
        if inliers > most_inliers: bestH, most_inliers, best_args = H, inliers, args

    return bestH

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')

    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    
    locs1, locs2, desc1, desc2, matches = np.load('BRIEF-ch.npy')
    H = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    
    # matches = matches[args]
    # plotMatches(im1, im2, matches, locs1, locs2)
    # displayPoints(im2, ret)
