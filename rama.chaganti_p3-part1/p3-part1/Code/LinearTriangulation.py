import numpy as np
import sys
from numpy import linalg as LA

# Function to perform Linear Triangulation to estimate 3D points from two camera views
def LinearTriangulation(K, C0, R0, Cseti, Rseti, x1set, x2set):
    """
    LinearTriangulation: Computes 3D points from two sets of 2D correspondences (x1set and x2set)
    observed from two different camera poses using linear triangulation.
    
    Parameters:
    - K: Intrinsic camera matrix (3x3).
    - C0: Camera center for the first camera pose (3x1).
    - R0: Rotation matrix for the first camera pose (3x3).
    - Cseti: Camera center for the second camera pose (3x1).
    - Rseti: Rotation matrix for the second camera pose (3x3).
    - x1set: DataFrame containing 2D points in the first image (ID, u, v).
    - x2set: DataFrame containing corresponding 2D points in the second image (ID, u, v).
    
    Returns:
    - Xset: Array of 3D points in homogeneous coordinates along with their IDs (Nx4).
    """
        
    Xset = []  # List to store the triangulated 3D points
    
    # Convert DataFrames to numpy arrays for easier manipulation
    x1set = x1set.to_numpy()
    x2set = x2set.to_numpy()

    
    # Calculate the projection matrices P1 and P2 with the form P = KR[I|-C]
    I = np.eye(3)  # Identity matrix (3x3)
    P1 = np.matmul(np.matmul(K, R0), np.concatenate((I, -C0), axis=1))  # Projection matrix for the first camera
    P2 = np.matmul(np.matmul(K, Rseti), np.concatenate((I, (-1) * Cseti), axis=1))  # Projection matrix for the second camera
    
    # Iterate over each pair of corresponding points (x1 in first view and x2 in second view)
    for x1, x2 in zip(x1set, x2set):
        ID = x1[0]  # Unique ID for the point
        u1 = x1[1]  # x-coordinate in the first image
        v1 = x1[2]  # y-coordinate in the first image
        
        u2 = x2[1]  # x-coordinate in the second image
        v2 = x2[2]  # y-coordinate in the second image
        
        # Construct matrix A for the linear triangulation system Ax=0
        # Each row in A is derived from the epipolar geometry constraint
        A = np.zeros((4, 4))
        A[0] = u1 * P1[2] - P1[0]
        A[1] = v1 * P1[2] - P1[1]
        A[2] = u2 * P2[2] - P2[0]
        A[3] = v2 * P2[2] - P2[1]
        
        # Solve Ax=0 using the eigenvector associated with the smallest eigenvalue of A^T A
        # Compute A^T * A
        ATA = np.matmul(A.T, A)
        
        # Eigen decomposition of A^T * A
        eigenvals, eigenvecs = LA.eig(ATA)
        
        # Find the smallest eigenvalue
        min_eigenval_idx = np.argmin(eigenvals)
        
        # Corresponding eigenvector gives the solution
        X = eigenvecs[:, min_eigenval_idx]
        
        # Normalize to make the point homogeneous
        X = X / X[3]
        
        # Append the triangulated 3D point with its ID to the list
        Xset.append([ID, X[0], X[1], X[2]])

    # Convert the list of points to a numpy array for easy manipulation
    Xset = np.array(Xset)

    return Xset  # Return the set of triangulated 3D points with their IDs
