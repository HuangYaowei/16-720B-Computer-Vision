"""
Homework4
Replace 'pass' by your implementation
"""

# Insert your package here
import sys

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import helper

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    assert(pts1.shape[0] >= 8)
    assert(pts1.shape[0] == pts2.shape[0])

    # Normalize points
    p1 = pts1/M
    p2 = pts2/M
    T = np.diag([1/M, 1/M, 1])

    # Create A matrix
    A = np.vstack([p1[:, 0]*p2[:, 0], p1[:, 0]*p2[:, 1], p1[:, 0],
                   p1[:, 1]*p2[:, 0], p1[:, 1]*p2[:, 1], p1[:, 1],
                   p2[:, 0], p2[:, 1], np.ones(p1.shape[0])])

    U, S, V = np.linalg.svd(A.T)
    F = V[-1, :].reshape(3, 3)

    F = helper._singularize(F)      # Singularize
    F = helper.refineF(F, p1, p2)   # Ensure non-singular
    F = T.T @ F @ T                 # Denormalize F

    return F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix
'''
def sevenpoint(pts1, pts2, M):
    assert(pts1.shape[0] == 7)
    assert(pts1.shape[0] == pts2.shape[0])

    # Normalize points
    p1 = pts1/M
    p2 = pts2/M
    T = np.diag([1/M, 1/M, 1])

    # Create A matrix
    A = np.vstack([p1[:, 0]*p2[:, 0], p1[:, 0]*p2[:, 1], p1[:, 0],
                   p1[:, 1]*p2[:, 0], p1[:, 1]*p2[:, 1], p1[:, 1],
                   p2[:, 0], p2[:, 1], np.ones(p1.shape[0])])

    U, S, V = np.linalg.svd(A.T)
    F1 = V[-1, :].reshape(3, 3)
    F2 = V[-2, :].reshape(3, 3)

    # Find coefficients
    fun = lambda a: np.linalg.det(a*F1 + (1 - a)*F2)
    a0 = fun(0)
    a1 = 2*(fun(1) - fun(-1))/3 - (fun(2) - fun(-2))/12
    a2 = 0.5*fun(1) + 0.5*fun(-1) - fun(0)
    a3 = (fun(2) - fun(-2))/12 - (fun(1) - fun(-1))/6

    # Find roots of the polynomial
    roots = np.roots([a3, a2, a1, a0])
    roots = np.real(roots[np.isreal(roots)])

    Farray = []
    for a in roots:
        F = a*F1 + (1 - a)*F2
        F = helper._singularize(F)      # Singularize
        F = helper.refineF(F, p1, p2)   # Ensure non-singular
        F = T.T @ F @ T                 # Denormalize F
        Farray.append(F)

    return Farray

'''
Q3.1: Compute the essential matrix E
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return K2.T @ F @ K1 

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error
'''
def triangulate(C1, pts1, C2, pts2):
    err = 0
    P = []

    for i in range(pts1.shape[0]):
        A = [pts1[i, 0] * C1[2, :] - C1[0, :],
             pts1[i, 1] * C1[2, :] - C1[1, :],
             pts2[i, 0] * C2[2, :] - C2[0, :],
             pts2[i, 1] * C2[2, :] - C2[1, :]]

        U, S, V = np.linalg.svd(np.asarray(A))
        w = V[-1, :]/V[-1, -1]
        P.append(w[:3])

        proj1 = C1 @ w
        proj2 = C2 @ w

        err += np.linalg.norm(proj1[:2]/proj1[-1] - pts1)**2 + np.linalg.norm(proj2[:2]/proj2[-1] - pts2)**2
        
    return np.asarray(P), err

'''
Q4.1: 3D visualization of the temple images
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pass

'''
Q5.1: RANSAC method
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
'''
def ransacF(pts1, pts2, M):
    # Replace pass by your implementation
    pass

'''
Q5.2: Rodrigues formula
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    pass

'''
Q5.2: Inverse Rodrigues formula
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    pass

'''
Q5.3: Rodrigues residual
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2
                Output: residuals, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Bundle adjustment
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass

if __name__ == '__main__':
    # Load data
    some_corresp = np.load('../data/some_corresp.npz')
    intrinsics = np.load('../data/intrinsics.npz')

    # Load images
    im1 = scipy.ndimage.imread('../data/im1.png')
    im2 = scipy.ndimage.imread('../data/im2.png')
    M = max(im1.shape)

    # Eight-point algorithm
    F = eightpoint(some_corresp['pts1'], some_corresp['pts2'], M)
    np.savez('q2_1', F=F, M=M)

    # Seven-point algorithm
    np.random.seed(4)
    r = np.random.randint(0, some_corresp['pts1'].shape[0], 7)
    F = sevenpoint(some_corresp['pts1'][r], some_corresp['pts2'][r], M)
    np.savez('q2_2', F=F[0], M=M, pts1=some_corresp['pts1'][r], pts2=some_corresp['pts2'][r])
    
    # Display on GUI
    helper.displayEpipolarF(im1, im2, F[0])
