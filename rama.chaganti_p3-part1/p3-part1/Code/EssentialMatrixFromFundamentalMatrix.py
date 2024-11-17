from numpy import linalg as LA
import numpy as np
# Function to compute the Essential Matrix from the Fundamental Matrix
def EssentialMatrixFromFundamentalMatrix(F, K):
    """
    EssentialMatrixFromFundamentalMatrix: Computes the Essential Matrix (E) from the Fundamental Matrix (F)
    and the camera intrinsic matrix (K).
    
    Parameters:
    - F: Fundamental matrix (3x3), relating corresponding points between two views in normalized image coordinates.
    - K: Intrinsic camera matrix (3x3), containing the intrinsic parameters of the camera.
    
    Returns:
    - new_E: Corrected Essential matrix (3x3) that enforces the constraints necessary for a valid Essential matrix.
    """

    # Transpose of the intrinsic matrix K
    K_T = K.T

    # Compute the initial Essential matrix E using E = K^T * F * K
    E = K_T @ F @ K

    # Apply Singular Value Decomposition (SVD) to E to enforce constraints for a valid Essential matrix
    U, S, Vh = LA.svd(E, full_matrices=True)
    # Essential matrix constraint: Enforce two singular values to be 1 and the third to be 0
    # This is because an Essential matrix has a rank of 2 and two equal non-zero singular values.
    S = np.diag(S)
    S[0, 0] = 1
    S[1, 1] = 1
    S[2, 2] = 0
    
    # Reconstruct the corrected Essential matrix by applying the modified singular values
    new_E = U @ S @ Vh
    
    return new_E  # Return the corrected Essential matrix
