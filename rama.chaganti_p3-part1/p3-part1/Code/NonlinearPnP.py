import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

# Function to perform Non-linear Perspective-n-Point (PnP) optimization to refine the camera pose
def NonlinearPnP(Xs, xs, K, Cnew, Rnew):
    """
    NonlinearPnP: Refines the camera pose (position and orientation) using non-linear optimization
    to minimize the reprojection error between observed 2D points and projected 3D points.
    
    Parameters:
    - Xs: DataFrame of 3D points in world coordinates (Nx4 with IDs).
    - xs: DataFrame of corresponding 2D points in image coordinates (Nx3 with IDs).
    - K: Intrinsic camera matrix (3x3).
    - Cnew: Initial guess for camera position (3x1).
    - Rnew: Initial guess for camera rotation matrix (3x3).
    
    Returns:
    - Copt: Optimized camera position (3x1).
    - Ropt: Optimized rotation matrix (3x3).
    """
    
    def reprojection_loss(x, Xset, xset, K):
        """
        Computes the reprojection error for the current camera pose estimate.
        
        Parameters:
        - x: Flattened array containing the camera position and rotation as a quaternion.
        - Xset: 3D points in world coordinates (Nx3).
        - xset: Corresponding 2D image points (Nx2).
        - K: Intrinsic camera matrix (3x3).
        
        Returns:
        - residuals: Flattened array of reprojection errors for each point.
        """
        
        # Extract the camera translation vector (position) from the optimization variable x
        C = x[:3][:, None]  # Camera position (3x1)
        # print(C)
        # Convert the quaternion (x[3:]) to a rotation matrix R (3x3)
        R = Rotation.from_quat(x[3:]).as_matrix()
        
        # print(f"in loss",R,C)
        # Construct the projection matrix P using the rotation and translation
        I = np.eye(3)
        P = np.matmul(np.matmul(K, R), np.concatenate((I, -C), axis=1))
        
        # Prepare the 3D points in homogeneous coordinates
        Xset = np.pad(Xset, ((0, 0), (0, 1)), constant_values=1).T  # [X, Y, Z, 1] format
        # print(Xset)
        # Project the 3D points Xset into the 2D image plane using the projection matrix P
        x_proj = np.matmul(P, Xset).T  # Projected 2D points in homogeneous coordinates [x, y, z]
        x_proj = x_proj / x_proj[:, 2, None]  # Normalize to get pixel coordinates [u, v, 1]
        x_proj = x_proj[:, :2]  # Extract [u, v] coordinates
        # print(f'Xset', Xset.dtype)
        # print(f'xxset', xset.dtype)
        # print(f'x_proj', x_proj.data)
        # Calculate the reprojection error as the difference between observed and projected points
        #print(f'xset', xset,'x_proj', x_proj)
        
        residuals = (xset - x_proj).ravel()  # Flatten the error array for least_squares
        # print(residuals)
        # print(f'residual before', residuals.dtype)
        # residuals = np.array(residuals, dtype = np.float64)

        # print(f'residual error:', residuals.dtype)
        # if not np.all(np.isfinite(residuals)):
        #     raise ValueError("Non-finite residuals computed in reprojection_loss.")
        return residuals
        
    # Convert initial rotation matrix Rnew to a quaternion representation
    r = Rotation.from_matrix(Rnew)
    quat = r.as_quat()
    
    # print(f"initial guess",Rnew,Cnew)
    # Initial parameters for optimization: flatten camera position and convert rotation to quaternion
    C_init = Cnew.flatten()  # Initial camera position as a 1D array (3,)
    q_init = quat  # Initial rotation as a quaternion (4,)
    x0 = np.concatenate([C_init, q_init])  # Combine position and quaternion for optimization
    if not np.all(np.isfinite(x0)):
        raise ValueError("Initial guess x0 contains non-finite values.")
    # print(quat,C_init)
    # Prepare the 3D points and 2D points for optimization
    X_data = Xs[:, 1:4]  # Extract [X, Y, Z] coordinates from the 3D points
    x_data = xs[:, 1:3] # Extract [u, v] coordinates from the 2D points
    
    #print(X_data,x_data)
    # Run non-linear optimization to minimize reprojection error
    # result = least_squares(reprojection_loss, x0, args=(X_data, x_data, K), verbose=2, method='lm')
    result = least_squares(reprojection_loss, x0, args=(X_data, x_data, K), method='trf',  # Trust-region reflective for handling nonlinearity
    # bounds=(lb, ub),  # Define parameter bounds
    loss='soft_l1',  # Robust loss for outliers and nonlinearity
    ftol=1e-8,
    xtol=1e-6,
    gtol=1e-8,
    max_nfev=5000
 ) 
    
    # Extract the optimized camera position and rotation matrix from the solution
    Copt = result.x[:3].reshape(3, 1)  # Optimized camera position (3x1)
    Ropt = Rotation.from_quat(result.x[3:]).as_matrix()  # Optimized rotation matrix (3x3)

    return Copt, Ropt  # Return the optimized camera position and rotation
