import numpy as np
import random
from tqdm import tqdm
from numpy import linalg as LA

# Function to calculate the reprojection error of a 3D point projected onto the image plane
def CalReprojErr(X, x, P):
    """
    CalReprojErr: Computes the reprojection error for a 3D point X when projected
    onto the image plane with a given camera matrix P.
    
    Parameters:
    - X: 3D point in homogeneous coordinates with an ID (4x1).
    - x: Observed 2D point in the image with an ID (3x1).
    - P: Projection matrix (3x4).
    
    Returns:
    - e: Reprojection error (scalar).
    """
    u = x[1]  # x-coordinate in the image
    v = x[2]  # y-coordinate in the image

    # Convert 3D point X to homogeneous coordinates without ID
    X_noID = np.concatenate((X[1:], np.array([1])), axis=0)
    
    # Extract rows of P for computation
    P1 = P[0, :]
    P2 = P[1, :]
    P3 = P[2, :]
    
    # Calculate reprojection error
    e = (u - np.matmul(P1, X_noID) / np.matmul(P3, X_noID))**2 + \
        (v - np.matmul(P2, X_noID) / np.matmul(P3, X_noID))**2
    
    return e

# Function to estimate the camera projection matrix using Linear Perspective-n-Point (PnP)
def LinearPnP(X, x, K):
    """
    LinearPnP: Computes the camera projection matrix (P) given a set of 3D points (X)
    and their corresponding 2D projections (x) using a linear approach.
    
    Parameters:
    - X: DataFrame of 3D points with IDs (Nx4).
    - x: DataFrame of corresponding 2D points with IDs (Nx3).
    - K: Intrinsic camera matrix (3x3).
    
    Returns:
    - P: Camera projection matrix (3x4).
    """
    X = X.to_numpy()
    x = x.to_numpy()

    # Construct the linear system A from the correspondences
    n = X.shape[0]
    A = np.zeros((2*n, 12))
    
    for i in range(n):
        X_i = np.concatenate((X[i, 1:], [1]))  # Convert to homogeneous coordinates
        u = x[i, 1]
        v = x[i, 2]
        
        # Fill A matrix based on the equations from the lecture
        A[2*i] = np.concatenate([np.zeros(4), -X_i, v*X_i])
        A[2*i + 1] = np.concatenate([X_i, np.zeros(4), -u*X_i])
    
    # Solve the linear system using Singular Value Decomposition (SVD)
    # Last column of V gives the solution for P
    _, _, V = np.linalg.svd(A)
    p = V[-1, :]  # Get last row of V
    P = p.reshape(3, 4)  # Reshape into 3x4 projection matrix
    
    # Recover the calibrated projection matrix
    P = np.linalg.inv(K) @ P
    
    return P

# Function to perform PnP using RANSAC to find the best camera pose with inliers
def PnPRANSAC(Xset, xset, K, M=2000, T=10):
    """
    PnPRANSAC: Performs Perspective-n-Point (PnP) with RANSAC to robustly estimate the
    camera pose (position and orientation) from 2D-3D correspondences.
    
    Parameters:
    - Xset: DataFrame of 3D points with IDs (Nx4).
    - xset: DataFrame of corresponding 2D points with IDs (Nx3).
    - K: Intrinsic camera matrix (3x3).
    - M: Number of RANSAC iterations (default: 2000).
    - T: Threshold for reprojection error to count as an inlier (default: 10).
    
    Returns:
    - Cnew: Estimated camera center (3x1).
    - Rnew: Estimated rotation matrix (3x3).
    - Inlier: List of inlier 3D points.
    """
    
    # List to store the largest set of inliers
    best_inliers = []
    # Total number of correspondences
    n_points = len(Xset)
    
    for i in tqdm(range(M)):
        # Randomly select 6 2D-3D pairs
        sample_indices = random.sample(range(n_points), 6)
                
        # Extract subsets of 3D and 2D points
        X_subset = Xset.iloc[sample_indices]
        x_subset = xset.iloc[sample_indices]

        # Estimate projection matrix P using LinearPnP with the selected points
        P = LinearPnP(X_subset, x_subset, K)
        
        # Calculate inliers by checking reprojection error for all points
        current_inliers = []
        for j in range(n_points):
            # Calculate reprojection error
            error = CalReprojErr(Xset.iloc[j].values, xset.iloc[j].values, P)
            
            # If error is below threshold T, consider it as an inlier
            if error < T:
                current_inliers.append(j)
        
        # Update inlier set if the current inlier count is the highest found so far
        if len(current_inliers) > len(best_inliers):
            best_inliers = current_inliers
            Pnew = P
    
    # Decompose Pnew to obtain rotation R and camera center C
    R = Pnew[:, :3]
    t = Pnew[:, 3]
    Cnew = -np.linalg.inv(R) @ t.reshape(3, 1)
    
    # Enforce orthogonality of R
    U, _, Vt = np.linalg.svd(R)
    Rnew = U @ Vt
    
    return Cnew, Rnew, best_inliers  # Return the estimated camera center, rotation matrix, and inliers
