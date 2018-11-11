import numpy as np

import helper
import submission as sub

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, p1, p2, R and P to q3_3.mat
'''
def findM2():
    # Load data
    some_corresp = np.load('../data/some_corresp.npz')
    intrinsics = np.load('../data/intrinsics.npz')

    # Compute fundamental and essential matrices
    F = sub.eightpoint(some_corresp['pts1'], some_corresp['pts2'], 640)
    E = sub.essentialMatrix(F, intrinsics['K1'], intrinsics['K2'])

    # Copmute camera 1 matrix
    M1 = np.concatenate([np.eye(3), np.zeros([3, 1])], axis=1)
    C1 = intrinsics['K1'] @ M1
    
    # Find the correct M2
    best_data = None
    best_err = np.finfo('float').max
    M2s = helper.camera2(E)
    
    for i in range(4):
        # Compute camera 2 matrix
        C2 = intrinsics['K2'] @ M2s[:, :, i]
        P, err = sub.triangulate(C1, some_corresp['pts1'], C2, some_corresp['pts2'])

        # Ensure all Z values are positive        
        if len(P[P[:, 2] > 0]) != P.shape[0]: continue

        # Save the correct data
        if err < best_err:
            best_err = err
            best_data = [M2s[:, :, i], C2, P]

    # Save the data
    np.savez('q3_3', M2=best_data[0], C2=best_data[1], P=best_data[2])

if __name__ == '__main__':
    findM2()
    