import numpy as np
import sys
import pandas as pd

from scipy.spatial.transform import Rotation
from BuildVisibilityMatrix import BuildVisibilityMatrix
from scipy.optimize import least_squares

# Function to perform Bundle Adjustment to optimize camera poses and 3D point positions
def BundleAdjustment(Call, Rall, Xall, K, n_cameras, n_points, camera_indices, point_indices, xall):
    """
    BundleAdjustment: Refines camera poses and 3D point positions to minimize the reprojection error
    for a set of cameras and 3D points using non-linear optimization.
    
    Parameters:
    - Call: List of initial camera positions (list of 3x1 arrays).
    - Rall: List of initial rotation matrices for each camera (list of 3x3 arrays).
    - Xall: DataFrame of 3D points with IDs (Nx4).
    - K: Intrinsic camera matrix (3x3).
    - sparseVmatrix: Sparse matrix for Jacobian sparsity pattern to speed up optimization.
    - n_cameras: Number of cameras.
    - n_points: Number of 3D points.
    - camera_indices: Indices indicating which camera observes each 2D point.
    - point_indices: Indices indicating which 3D point corresponds to each 2D point.
    - xall: Array of observed 2D points in image coordinates.
    
    Returns:
    - CoptAll: List of optimized camera positions (3x1 arrays).
    - RoptAll: List of optimized rotation matrices (3x3 arrays).
    - XoptAll: DataFrame of optimized 3D points with IDs.
    """
    
    def reprojection_loss(x, n_cameras, n_points, camera_indices, point_indices, xall, K):
        """
        Computes the reprojection error for the current estimates of camera poses and 3D points.
        
        Parameters:
        - x: Flattened array containing all camera positions, orientations (as quaternions), and 3D points.
        - n_cameras: Number of cameras.
        - n_points: Number of 3D points.
        - camera_indices: Indices indicating which camera observes each 2D point.
        - point_indices: Indices indicating which 3D point corresponds to each 2D point.
        - xall: Observed 2D points (Nx2).
        - K: Intrinsic camera matrix (3x3).
        
        Returns:
        - residuals: Flattened array of reprojection errors for all points across all cameras.
        """
        
        I = np.eye(3)  # Identity matrix for projection matrix construction
        
        # First camera pose (fixed as the reference)
        C0 = np.zeros((3, 1))  # Assume the first camera position is at the origin
        R0 = np.eye(3)  # Assume the first camera has no rotation
        P0 = np.matmul(np.matmul(K, R0), np.concatenate((I, -C0), axis=1))  # Projection matrix for the first camera
        
        Ps = [P0]  # Initialize list of projection matrices with the first camera's projection matrix
        
        # Reconstruct the remaining camera poses from the optimization variable `x`
        for i in range(n_cameras - 1):
            C = x[7*i:7*i + 3].reshape(3, 1)  # Extract camera position
            q = x[7*i + 3:7*i + 7]  # Extract quaternion
            R = Rotation.from_quat(q).as_matrix()  # Convert quaternion to rotation matrix
            P = np.matmul(np.matmul(K, R), np.concatenate((I, -C), axis=1))  # Construct projection matrix

  
            Ps.append(P)

 
        Pall = np.array([Ps[int(idx)] for idx in camera_indices-1])
        

        X = x[7 * (n_cameras - 1):].reshape((-1, 3))
        
      
        Xall = np.array([np.pad(X[int(idx)], (0, 1), constant_values=1)[:, None] for idx in point_indices])

        x_proj = np.squeeze(np.matmul(Pall, Xall))  
        x_proj = x_proj / x_proj[:, 2, None]
        x_proj = x_proj[:, :2]  
        
        reprojection_error = (xall - x_proj).ravel()
      
        
        return reprojection_error
    
    
    # print("\n Running BA.....")

  
    init_x = np.array([])
    for idx, (Ci, Ri) in enumerate(zip(Call, Rall)):
                
       
        qi = Rotation.from_matrix(Ri).as_quat() 
        
        if idx == 0:
          
            continue  
        else:
            # Append other cameras' parameters
            init_x = np.concatenate((init_x, Ci.flatten()))  # Add camera position
            init_x = np.concatenate((init_x, qi))  # Add quaternion
    
    
    X_init = Xall[:, 1:4]
    
    point_ids = Xall[:, 0]
   
    
   
    init_x = np.append(init_x, X_init.flatten(), axis=0)
   
    # print(f"init_x size: {init_x.size}")
 
    
    A = BuildVisibilityMatrix(n_cameras, n_points, camera_indices, point_indices)
    A = A[:, :init_x.size]

    
    
    

    # Perform bundle adjustment using non-linear least squares optimization
    result = least_squares(reprojection_loss, init_x, args=(n_cameras, n_points, camera_indices, point_indices, xall, K),
                         jac_sparsity=A, x_scale='jac', method='trf')

    # Extract optimized camera poses from the solution
    CoptAll = [np.zeros((3, 1))]  # First camera remains at origin
    RoptAll = [np.eye(3)]  # First camera remains unrotated
    
    for i in range(n_cameras - 1):
        Copt = result.x[7*i:7*i + 3].reshape(3, 1)
        qopt = result.x[7*i + 3:7*i + 7]
        Ropt = Rotation.from_quat(qopt).as_matrix()
        CoptAll.append(np.array(Copt))
        RoptAll.append(np.array(Ropt))
        
    # Extract optimized 3D points and combine with IDs
    Xopt = result.x[7*(n_cameras-1):].reshape((-1, 3))
    XoptAll = pd.DataFrame(np.column_stack((point_ids, Xopt)), columns=['ID', 'X', 'Y', 'Z']).to_numpy()
    
    return CoptAll, RoptAll, XoptAll  # Return optimized camera positions, rotation matrices, and 3D points
