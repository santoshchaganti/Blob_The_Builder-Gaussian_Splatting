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
        
    Xset = []
    
    # Convert DataFrames to numpy arrays
    x1set = x1set.to_numpy()
    x2set = x2set.to_numpy()
    
    # Reshape camera centers
    C0 = C0.reshape(3, 1)
    Cseti = Cseti.reshape(3, 1)
    
    # Calculate projection matrices
    I = np.eye(3)
    P1 = np.matmul(K, np.matmul(R0, np.hstack((I, -C0))))
    P2 = np.matmul(K, np.matmul(Rseti, np.hstack((I, -Cseti))))
    
    # Get individual rows of projection matrices
    p1, p2, p3 = P1
    p1_cap, p2_cap, p3_cap = P2
    
    # Process each pair of corresponding points
    for x1, x2 in zip(x1set, x2set):
        ID = x1[0]  # Get point ID
        
        # Extract coordinates
        x = x1[0]   # u1
        y = x1[1]   # v1
        x_cap = x2[0]   # u2
        y_cap = x2[1]   # v2
        
        # Construct constraints matrix A
        A = []
        A.append((y * p3) - p2)
        A.append((x * p3) - p1)
        A.append((y_cap * p3_cap) - p2_cap)
        A.append((x_cap * p3_cap) - p1_cap)
        
        A = np.array(A).reshape(4, 4)
        
        # Solve using SVD
        _, _, v = np.linalg.svd(A)
        X = v[-1, :]
        X = X / X[-1]  # Normalize homogeneous coordinates
        
        # Append point ID and 3D coordinates
        Xset.append([ID, X[0], X[1], X[2]])
    
    # Convert to numpy array
    Xset = np.array(Xset)
    
    return Xset
