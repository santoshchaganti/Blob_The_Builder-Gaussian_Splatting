import numpy as np
from scipy.optimize import least_squares

# Function to perform Non-linear Triangulation to refine 3D points given initial estimates
def NonlinearTriangulation(K, C0, R0, Cseti, Rseti, x1set, x2set, X0):
    """
    NonlinearTriangulation: Refines the initial estimates of 3D points by minimizing the reprojection error
    through non-linear optimization.
    
    Parameters:
    - K: Intrinsic camera matrix (3x3).
    - C0: Camera center for the first camera pose (3x1).
    - R0: Rotation matrix for the first camera pose (3x3).
    - Cseti: Camera center for the second camera pose (3x1).
    - Rseti: Rotation matrix for the second camera pose (3x3).
    - x1set: DataFrame containing 2D points in the first image (ID, u, v).
    - x2set: DataFrame containing corresponding 2D points in the second image (ID, u, v).
    - X0: Initial estimates of 3D points, including point IDs (Nx4).
    
    Returns:
    - Xopt: Optimized 3D points with IDs (Nx4).
    """
    
    def reprojection_loss(x, Ps, xsets):
        """
        Computes the reprojection error between the observed 2D points and the projected 3D points.
        
        Parameters:
        - x: Flattened array of 3D point coordinates to be optimized (1D array).
        - Ps: List of camera projection matrices (one for each camera view).
        - xsets: List of 2D point sets (one set of points for each camera view).
        
        Returns:
        - residuals: Flattened array of reprojection errors for all points in both views.
        """
        
        # Reshape the 1D array of 3D points to an Nx3 matrix and convert to homogeneous coordinates
        X = x.reshape(-1, 3)
        # Transform to homogeneous [X, Y, Z, 1]
        X_homog = np.hstack((X, np.ones((X.shape[0], 1))))

        # Calculate reprojection error for each camera and corresponding 2D points
        for idx, (Pi, xi) in enumerate(zip(Ps, xsets)):
            # Project the 3D points X onto the image plane of the current camera
            x_proj_homog = Pi @ X_homog.T
            # Projected 2D points in homogeneous coordinates [x, y, z]
            x_proj_homog = x_proj_homog.T
            # Normalize to get pixel coordinates [u, v, 1]
            x_proj = x_proj_homog[:, :2] / x_proj_homog[:, 2:]
            # Extract [u, v] coordinates
            x_obs = xi[:, :2]
            
            # Calculate the reprojection error as the difference between observed and projected points
            if idx == 0:
                # Flatten the error for the first camera view
                error = (x_obs - x_proj).ravel()
            else:
                # Concatenate errors for both views
                error = np.concatenate([error, (x_obs - x_proj).ravel()])
        return error
    
    # Extract point IDs and initial guess for 3D points (ignoring the IDs for optimization)
    point_ids = X0[:, 0]
    X_init = X0[:, 1:].astype(float)
    # Flatten the initial 3D points to 1D array for optimization
    x0 = X_init.ravel()
    
    # Construct camera projection matrices for each view using intrinsic and extrinsic parameters
    ## P = K * R * [I | -C]
    # Identity matrix (3x3)
    I = np.eye(3)
    # Projection matrix for the first camera
    P1 = K @ R0 @ np.hstack((I, -C0))
    # Projection matrix for the second camera
    P2 = K @ Rseti @ np.hstack((I, -Cseti))
    # List of projection matrices for both views
    Ps = [P1, P2]
    
    # Prepare sets of 2D points for each camera view
    # Extract [u, v] coordinates from the first image's points
    x1 = x1set.iloc[:, 1:].values
    # Extract [u, v] coordinates from the second image's points
    x2 = x2set.iloc[:, 1:].values
    # List of 2D point sets for both views
    xsets = [x1, x2]
    
    # Perform non-linear optimization to minimize reprojection error
    result = least_squares(reprojection_loss, x0, args=(Ps, xsets), method='lm')
    # Reshape optimized 3D points to Nx3 matrix
    X_opt = result.x.reshape(-1, 3)
    
    # Combine and return optimized 3D points with their corresponding IDs
    return np.column_stack((point_ids, X_opt))
