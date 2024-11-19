import numpy as np
from scipy.sparse import lil_matrix

# Function to create a visibility matrix from inlier data
def GetVmatrix(All_Inlier):
    """
    GetVmatrix: Extracts and combines u, v coordinates, Point IDs, and Camera IDs from inlier data.
    
    Parameters:
    - All_Inlier: DataFrame of inliers, where each row contains PointID, (u, v) coordinates, and CameraID.
    
    Returns:
    - Vmatrix: Combined matrix of u, v coordinates, Point IDs, and Camera IDs.
    """
      
    # Extract each component from the array
    u = All_Inlier.iloc[:, 1].values  # u coordinates
    v = All_Inlier.iloc[:, 2].values  # v coordinates
    point_ids = All_Inlier.iloc[:, 0].values  # Point IDs
    camera_ids = All_Inlier.iloc[:, 3].values  # Camera IDs

    # Concatenate u, v, PointID, and CameraID into a single matrix
    Vmatrix = np.column_stack((u, v, point_ids, camera_ids))
    
    return Vmatrix

# Function to build the visibility matrix for bundle adjustment
def BuildVisibilityMatrix(n_cameras, n_points, camera_indices, point_indices):
    """
    BuildVisibilityMatrix: Constructs a sparse visibility matrix for bundle adjustment.
    This matrix indicates which parameters affect each observation in the optimization process.
    
    Parameters:
    - n_cameras: Total number of cameras.
    - n_points: Total number of 3D points.
    - camera_indices: Array of indices indicating which camera observes each 2D point.
    - point_indices: Array of indices indicating which 3D point corresponds to each 2D point.
    
    Returns:
    - A: Sparse visibility matrix in lil_matrix format.
    """
    
    # Calculate the number of observations (2D points), each observation contributes two rows (u and v)
    n_observations = len(camera_indices)
    m = 2 * n_observations  # Total rows in visibility matrix (2 per observation)
    
    # Calculate the number of parameters (unknowns) in the optimization
    # Each camera has 7 parameters (3 for translation, 4 for rotation as quaternion)
    # Each 3D point has 3 parameters (X, Y, Z)
    n = 7 * n_cameras + 3 * n_points  # Total columns in visibility matrix
    
    # Initialize a sparse matrix in 'list of lists' (lil) format for efficient row operations
    A = lil_matrix((m, n), dtype=int)
    
    # Create an array of observation indices
    i = np.arange(n_observations)
    
    # Fill in the visibility matrix for camera parameters
    for s in range(7):
        A[2*i, camera_indices * 7 + s] = 1
        A[2*i + 1, camera_indices * 7 + s] = 1
    
    # Fill in the visibility matrix for 3D point parameters
    for s in range(3):
        A[2*i, 7*n_cameras + point_indices * 3 + s] = 1
        A[2*i + 1, 7*n_cameras + point_indices * 3 + s] = 1

    return A  # Return the sparse visibility matrix
