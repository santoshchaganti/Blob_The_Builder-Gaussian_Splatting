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
    x1DF = x1DF.to_numpy()
    x2DF = x2DF.to_numpy()

    # Step 1: Normalize Points
    def normalize_points(points):
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        mean_dist = np.mean(np.sqrt(np.sum(centered_points**2, axis=1)))
        scale = np.sqrt(2) / mean_dist
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        normalized_points = np.dot(T, np.vstack((points.T, np.ones((1, points.shape[0])))))
        return T, normalized_points.T

    T1, x1_norm = normalize_points(x1DF)
    T2, x2_norm = normalize_points(x2DF)

    # Step 2: Construct the Linear System (Matrix A)
    A = []
    for i in range(x1_norm.shape[0]):
        x1, y1 = x1_norm[i, :2]
        x2, y2 = x2_norm[i, :2]
        A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
    A = np.array(A)

    # Step 3: Solve Using SVD
    U, S, Vt = LA.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Step 4: Enforce Rank-2 Constraint on F
    U, S, Vt = LA.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))

    # Step 5: Scaling - Ensure the last element of F is 1
    F = F / F[2, 2]

    # Step 6: Denormalize the Fundamental Matrix
    F = np.dot(T2.T, np.dot(F, T1))
    

    return F