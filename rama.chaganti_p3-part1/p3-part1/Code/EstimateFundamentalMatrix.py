import numpy as np
from numpy import linalg as LA
import math
import sys


def EstimateFundamentalMatrix(x1DF, x2DF):
    """
    Estimates the Fundamental matrix (F) between two sets of points using the 8-point algorithm,
    with normalization for numerical stability.

    Args:
        x1DF (DataFrame): Source image points (x, y).
        x2DF (DataFrame): Target image points (x, y).

    Returns:
        F (ndarray): The estimated 3x3 Fundamental matrix.
    """
    # Convert input data from DataFrames to numpy arrays for easier manipulation
    x1DF = x1DF.to_numpy()
    x2DF = x2DF.to_numpy()
    
    # Step 1: Normalization - Improves numerical stability by normalizing points to a common scale
    # Compute centroids of the points in each image
    x1_centroid = np.mean(x1DF, axis=0)
    x2_centroid = np.mean(x2DF, axis=0)
    # Shift points to have zero mean (centering) by subtracting the centroid
    x1_centered = x1DF - x1_centroid
    x2_centered = x2DF - x2_centroid
    
    x1_dist = np.sqrt(np.sum(x1_centered**2, axis=1)).mean()
    x2_dist = np.sqrt(np.sum(x2_centered**2, axis=1)).mean()
    # Compute scaling factors to make the mean distance of points from the origin equal to sqrt(2)
    s1 = np.sqrt(2) / x1_dist
    s2 = np.sqrt(2) / x2_dist
    # Construct the normalization transformation matrices for both images
    T1 = np.array([
        [s1, 0, -s1*x1_centroid[0]],
        [0, s1, -s1*x1_centroid[1]], 
        [0, 0, 1]
    ])

    T2 = np.array([
        [s2, 0, -s2*x2_centroid[0]],
        [0, s2, -s2*x2_centroid[1]],
        [0, 0, 1]
    ])

    # Convert points to homogeneous coordinates
    x1_homog = np.column_stack((x1DF, np.ones(len(x1DF))))
    x2_homog = np.column_stack((x2DF, np.ones(len(x2DF))))

    # Apply normalization
    x1_norm = (T1 @ x1_homog.T).T
    x2_norm = (T2 @ x2_homog.T).T

    # Step 2: Construct the Linear System (A) for Estimating F
    # Set up a linear system A*f = 0, where f is the vector form of the matrix F
    n = len(x1_norm)
    A = np.zeros((n, 9))
    # Construct the first row of A using the epipolar constraint x2^T * F * x1 = 0
    # Repeat for all other points to build the full matrix A
    for i in range(n):
        x1, y1 = x1_norm[i,0], x1_norm[i,1]
        x2, y2 = x2_norm[i,0], x2_norm[i,1]
        A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        
    # Step 3: Solve the Linear System Using SVD
    # Find the solution to A*f = 0 by taking the SVD of A and using the right singular vector corresponding to the smallest singular value
    # Hint: u, s, vh = LA.svd(A)
    U, S, Vh = LA.svd(A, full_matrices=True)
    # F is the last column of V (smallest singular value)
    F = Vh[-1, :]
    # Reshape F into a 3x3 matrix
    F = F.reshape(3, 3)

    # Step 4: Enforce Rank-2 Constraint on F
    # The Fundamental matrix must be rank-2, so we set the smallest singular value of F to zero
    Ur, Sr, Vr = LA.svd(F, full_matrices=True)
    Sr = np.diag(Sr)
    # Set the smallest singular value to zero
    Sr[2, 2] = 0
    # Reconstruct F with rank-2 constraint
    F = Ur @ Sr @ Vr

    # Step 5: Scaling - Ensure the last element of F is 1 for consistency
    F = F / F[2, 2]

    # Step 6: Denormalize the Fundamental Matrix
    # Transform F back to the original coordinate system using the inverse of the normalization transformations
    F = T2.T @ F @ T1
    
    return F  # Return the estimated Fundamental matrix