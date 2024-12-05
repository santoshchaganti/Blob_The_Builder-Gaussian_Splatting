import numpy as np

def DisambiguateCameraPose(Cset, Rset, Xset):
    """
    DisambiguateCameraPose: Determines the correct camera pose (position and orientation)
    from a set of candidate poses based on the positive depth criterion.
    
    Parameters:
    - Cset: List of candidate camera positions (each 3x1 array).
    - Rset: List of candidate rotation matrices (each 3x3 array).
    - Xset: List of sets of 3D points for each candidate camera pose (each set of Nx4 arrays).
    
    Returns:
    - C: The correct camera position (3x1).
    - R: The correct rotation matrix (3x3).
    - X: The set of 3D points corresponding to the correct camera pose.
    - max_index: Index of the correct camera pose in the input lists.
    """
    # Reference pose: first camera at origin with identity rotation
    Rset0 = np.eye(3)
    Cset0 = np.zeros((3, 1))
    count_list = []  # List to store the count of points with positive depth for each candidate pose

    # Iterate over each candidate pose
    for i, (Cseti, Rseti, Xseti) in enumerate(zip(Cset, Rset, Xset)):
        count = 0  # Initialize count of points with positive depth
        
        for Xi in Xseti:
            # Extract 3D point in world coordinates
            X_world = Xi[1:4]  # Skip ID, consider [X, Y, Z] coordinates
            #print(X_world.shape)
            X_world_homo = np.hstack((X_world, [1]))  # Convert to homogeneous
            #print(X_world_homo.shape)
            # Compute the depth in the candidate camera's coordinate system
            X_cam_candidate = Rseti @ (X_world_homo[:3] - Cseti.flatten())
            depth_candidate = X_cam_candidate[2]  # Z-coordinate

            # Compute the depth in the reference camera's coordinate system
            X_cam_ref = Rset0 @ (X_world_homo[:3] - Cset0.flatten())
            depth_ref = X_cam_ref[2]  # Z-coordinate

            # Check positive depth for both cameras
            if depth_candidate > 0 and depth_ref > 0:
                count += 1

        # Store count of positive depth points for this pose
        count_list.append(count)

    # Find the index of the pose with the maximum count of points with positive depth
    max_index = np.argmax(count_list)

    # Select the correct pose based on the index
    C = Cset[max_index]
    R = Rset[max_index]
    X = Xset[max_index]

    return C, R, X, max_index
