import cv2
import numpy as np

from BRIEF import briefLite, briefMatch

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
    for i in range(p2.shape[1]):
        A.append([-p2[0, i], -p2[1, i], -1, 0, 0, 0, p2[0, i]*p1[0, i], p1[0, i]*p2[1, i], p1[0, i]])
        A.append([0, 0, 0, -p2[0, i], -p2[1, i], -1, p2[0, i]*p1[1, i], p1[1, i]*p2[1, i], p1[1, i]])

    U, S, V = np.linalg.svd(np.asarray(A))
    H2to1 = V[-1, :].reshape(3, 3)/V[-1, -1]

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
    p1 = locs1[matches[:, 0], 0:2].swapaxes(0, 1)
    p2 = locs2[matches[:, 1], 0:2].swapaxes(0, 1)

    # RANSAC
    bestH, most_inliers = None, 0
    for i in range(num_iter):
        # Pick 4 random matched points and calculate homography
        random_indices = np.random.randint(n_matches, size=4)
        H = computeH(p1[:, random_indices], p2[:, random_indices])

        # Compute H1to2
        H_inv = np.linalg.inv(H)
        H_inv = H_inv/H_inv[-1, -1]

        # Back project
        proj = np.matmul(H_inv, np.vstack((p1, [1]*n_matches)))
        proj = proj/proj[-1, :]
        
        # Calculate back projection error and number of inliers within tolerance
        distances = np.linalg.norm(proj.T - np.vstack((p2, [1]*n_matches)).T, axis=1)
        inliers = len(distances[distances<tol])
        if inliers > most_inliers: bestH, most_inliers = H, inliers

    return bestH

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    
    H = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    print(H)
    