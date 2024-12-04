from numpy import linalg as LA
import numpy as np

# # Function to ensure the rotation matrix R has a positive determinant
# def CheckDet(R, C):
#     """
#     CheckDet: Adjusts the rotation matrix R and translation vector C if the determinant of R is negative.
#     This is done to ensure that R represents a valid rotation matrix with a determinant of +1.
    
#     Parameters:
#     - R: Rotation matrix (3x3).
#     - C: Translation vector (3x1).
    
#     Returns:
#     - Adjusted R and C such that det(R) >= 0.
#     """
#     # If the determinant of R is -1, invert both R and C
#     if LA.det(R) < 0:
#         R = -R
#         C = -C

#     return R, C

# Function to extract possible camera poses from the Essential Matrix
def ExtractCameraPose(E):
    """
    ExtractCameraPose: Extracts four possible camera poses (rotation and translation pairs) from
    the Essential Matrix (E) using Singular Value Decomposition (SVD).
    
    Parameters:
    - E: Essential matrix (3x3).
    
    Returns:
    - Cset: List of four possible camera translation vectors (3x1).
    - Rset: List of four possible camera rotation matrices (3x3).
    """

    # Perform Singular Value Decomposition (SVD) on the Essential Matrix E
    U, S, Vh = LA.svd(E)

    # Define the rotation matrix W, which is used to construct possible rotations
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Compute the four possible camera poses (two rotations and two translations)
    R1 = np.dot(U, np.dot(W, Vh))
    R2 = np.dot(U, np.dot(W, Vh))
    R3 = np.dot(U, np.dot(W.T, Vh))
    R4 = np.dot(U, np.dot(W.T, Vh))
    
    C1 = U[:, 2]
    C2 = -U[:, 2]
    C3 = U[:, 2]
    C4 = -U[:, 2]
    
    # Collect the four possible camera poses (rotations and translations)
    Cset = [C1, C2, C3, C4]  # List of possible translation vectors
    Rset = [R1, R2, R3, R4]  # List of possible rotation matrices
    
    for i in range(len(Rset)):
        if (np.linalg.det(Rset[i]) < 0):
            Rset[i] = -Rset[i]
            Cset[i] = -Cset[i]

    return Cset, Rset  # Return sets of possible translations and rotations
