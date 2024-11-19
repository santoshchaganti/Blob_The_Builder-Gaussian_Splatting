import numpy as np

# Function to disambiguate the correct camera pose from multiple candidates
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
    
    # Reference pose (assuming first camera is at origin with identity rotation)
    Rset0 = np.eye(3)
    Cset0 = np.zeros((3, 1))
    countList = []  # List to store the count of points with positive depth for each candidate pose

    # Iterate over each candidate pose
    for Cseti, Rseti, Xseti in zip(Cset, Rset, Xset):       
        # Extract the third row of the rotation matrix (Z-axis direction)
        r3 = Rseti[2]
        r30 = Rset0[2]
        
        count = 0  # Initialize count of points with positive depth

        # For each 3D point in the current candidate pose
        for Xi in Xseti:
            # Transpose the 3D point to align with calculation
            # Convert 3D point to a column vector [X, Y, Z]
            X = Xi[1:].reshape(3,1)
            
            # Check if the point is in front of both the candidate and reference cameras
            # The depth check is performed in the camera coordinate system
            depth1 = r3 @ (X - Cseti)
            depth2 = r30 @ (X - Cset0)
            
            # If true: Increment count if the point has positive depth in both systems
            if depth1 > 0 and depth2 > 0:
                count += 1
                
        # Store the count of positive depth points for the current pose
        countList.append(count)
    
    # Find the candidate pose with the maximum count of points with positive depth
    max_index = np.argmax(countList)

    # Select the pose with the highest positive depth count as the correct pose
    C = Cset[max_index]
    R = Rset[max_index] 
    X = Xset[max_index]

    return C, R, X, max_index
