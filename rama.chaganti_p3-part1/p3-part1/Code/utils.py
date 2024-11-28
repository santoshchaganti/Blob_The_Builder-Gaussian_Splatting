import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

# Define Colors For Terminal Print
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

"""
Assign Unique IDs to feature points across datasets from any number of matching files.
"""
def IndexAllFeaturePoints(*files):
    """
    Processes multiple feature point matching files, assigns unique IDs to each feature point,
    and ensures that matching points across files have consistent IDs.

    Args:
        *files (str): Paths to the feature matching files (e.g., matching1.txt, matching2.txt, ..., matchingN.txt).
                      Each file should contain feature points for matching across images.

    Returns:
        list: A list of pandas DataFrames, each representing the processed data
              for each matching file with unique feature IDs assigned.
              The columns include:
              - ID: Unique identifier for each feature point.
              - Feature coordinates and other attributes, as extracted from the files.
    """

    dataframes = []  # To store each DataFrame for further processing and comparison
    count = -1       # Initialize feature count for assigning unique IDs

    # Loop through each file provided in args
    for file_index, file in enumerate(files):
        print(f"\nProcessing matching files from Image {file_index + 1}...")

        # Read txt file and skip the header
        with open(file, 'r') as fin:
            data = fin.read().splitlines(True)
        Newdata = data[1:]

        # Parse the lines into a list of feature points
        rawList = []
        for line in Newdata:
            raw = line.split(' ')
            rawList.append(raw)

        # Create DataFrame for feature points and process duplicates and empty cells
        rawDF = pd.DataFrame(rawList)
        matchDF = (
            rawDF
            .sort_values(by=[4, 5, 0])  # Sort by relevant columns for duplicate removal
            .drop_duplicates(subset=[4, 5], keep='last')  # Remove duplicate feature points
            .fillna(100000)  # Fill empty cells with placeholder
            .replace([' ', '\n'], 100000, regex=True)  # Replace any blanks/newlines
            .astype(float)  # Convert to float for further processing
            .replace(100000, '')  # Restore empty cells
        )
        matchDF.insert(0, "ID", '')  # Insert ID column, initially empty

        # Assign FeatureID by comparing with previously processed dataframes
        for index, row in matchDF.iterrows():
            found = False
            # Check against all previous DataFrames
            for prev_index, prev_df in enumerate(dataframes):
                # Adjust the column index based on each file's feature location
                same = prev_df.loc[
                    (prev_df.iloc[:, 1] == row[1]) &
                    (prev_df.iloc[:, 4] == row[4]) &
                    (prev_df.iloc[:, 5] == row[5])
                ]
                if not same.empty:
                    matchDF.at[index, 'ID'] = same['ID'].values[0]
                    found = True
                    break
            if not found:
                count += 1
                matchDF.at[index, 'ID'] = count

        dataframes.append(matchDF)  # Add the current DataFrame to the list

    # Write each DataFrame to a new text file with the updated IDs
    for i, df in enumerate(dataframes):
        df.to_csv(f'../Data/new_matching{i+1}.txt', header=None, index=None, sep=' ')
    # Define a list of file paths for matching files
    matching_files = [f'../Data/matching{i+1}.txt' for i in range(5)]  # Adjust range for more or fewer files as needed

    return dataframes


#######################################################################################
"""
The process_calib_matrix function reads a calibration file to construct and 
return the intrinsic camera matrix 𝐾, which contains essential parameters 
for camera calibration, such as focal lengths and the principal point.
"""
def process_calib_matrix(calib_file):
    # Read, clean, and parse the calibration data into a 3x3 matrix
    with open(calib_file, 'r') as fin:
        # Read all lines in the file and store them in a list called 'data'
        data = fin.read().splitlines(True)
    
    # Process the first line to remove '[' and ';', then split it into individual components
    line0 = data[0].replace('[', '').replace(';', ' ').split(' ')
    
    # Process the second line to remove ';', then split it into individual components
    line1 = data[1].replace(';', ' ').split(' ')
    
    # Process the third line to remove ']', then split it into individual components
    line2 = data[2].replace(']', '').split(' ')

    # Construct the intrinsic camera matrix K using specific elements from each processed line
    # Each element is accessed by index and converted to a 3x3 numpy array of floats
    K = np.array([
        [line0[2], line0[3], line0[4]],  # Extract the 3 relevant values from the first row
        [line1[5], line1[6], line1[7]],  # Extract the 3 relevant values from the second row
        [line2[5], line2[6], line2[7]]   # Extract the 3 relevant values from the third row
    ])
    
    # Convert all elements in K to floats, as they are initially read as strings
    K = K.astype(float)
    
    # Return the intrinsic matrix K
    return K

#########################################################################################################
def DrawMatches(imgPath, sourceIdx, targetIdx, sourceAllPts, targetAllPts, sourceInPts, targetInPts, outputPath):
    """
    Draws and saves images showing matched keypoints between two images, highlighting all
     matches and inlier matches.

    Args:
        imgPath (str): Path to the directory containing the source and target images.
        sourceIdx (int): Index of the source image file (expects a filename like "sourceIdx.jpg").
        targetIdx (int): Index of the target image file (expects a filename like "targetIdx.jpg").
        sourceAllPts (array): Array of all keypoints in the source image.
        targetAllPts (array): Array of all keypoints in the target image.
        sourceInPts (array): Array of inlier keypoints in the source image.
        targetInPts (array): Array of inlier keypoints in the target image.
        outputPath (str): Path to save the output images showing matches.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    Path(outputPath).mkdir(parents=True, exist_ok=True)

    # Load source and target images
    srcImg = cv2.imread(os.path.join(imgPath, f"{sourceIdx}.jpg"))
    tgtImg = cv2.imread(os.path.join(imgPath, f"{targetIdx}.jpg"))

    # Convert all matching points in the source and target images into KeyPoint objects
    sourceAllKeyPts = [cv2.KeyPoint(float(pt[0]), float(pt[1]), size=1.0) for pt in sourceAllPts]
    targetAllKeyPts = [cv2.KeyPoint(float(pt[0]), float(pt[1]), size=1.0) for pt in targetAllPts]

    # Convert inlier matching points into KeyPoint objects for source and target images
    sourceInKeyPts = [cv2.KeyPoint(float(pt[0]), float(pt[1]), size=1.0) for pt in sourceInPts]
    targetInKeyPts = [cv2.KeyPoint(float(pt[0]), float(pt[1]), size=1.0) for pt in targetInPts]

    # Create match objects for all points and inlier points
    Allmatches = [cv2.DMatch(idx, idx, 1.0) for idx in range(len(sourceAllKeyPts))]
    Inmatches = [cv2.DMatch(idx, idx, 1.0) for idx in range(len(sourceInKeyPts))]

    # Draw all matches with red lines (indicating all correspondences)
    AlloutImg = cv2.drawMatches(srcImg, sourceAllKeyPts, tgtImg, targetAllKeyPts, Allmatches, None,
                                matchColor=(0, 0, 255), singlePointColor=(0, 0, 255), flags=0)

    # Draw inlier matches with green lines (indicating matches that passed RANSAC filtering)
    InoutImg = cv2.drawMatches(srcImg, sourceInKeyPts, tgtImg, targetInKeyPts, Inmatches, None,
                               matchColor=(0, 255, 0), singlePointColor=(0, 255, 0), flags=0)

    # Save the images showing all matches and inlier matches
    cv2.imwrite(os.path.join(outputPath, f"{sourceIdx}_and_{targetIdx}_all.jpg"), AlloutImg)
    cv2.imwrite(os.path.join(outputPath, f"{sourceIdx}_and_{targetIdx}_inliers.jpg"), InoutImg)

#######################################################################################
def ParseKeypoints(file, sourceIdx, targetIdx):
    """
    Extracts matching keypoints between specified camera indices from a given file.

    Args:
        file (str): Path to the file containing keypoint matching data.
        source_idx (int): Index of the source camera.
        target_idx (int): Index of the target camera.

    Returns:
        pd.DataFrame: DataFrame with columns for keypoint ID, camera indices, and coordinates.
    """
    # Read txt file
    with open(file,'r') as fin:
        data = fin.read().splitlines(True)
    
    pairList = []
    for line in data:
        pair = line.split(' ')
        ID = int(pair[0])
        current_x = float(pair[5])
        current_y = float(pair[6])
        # Get target pair xy
        for i in range((int(float(pair[1]))-1)):
            ImgIdx = int(float(pair[7 + 3*i]))
            if ImgIdx == targetIdx:
                target_x = float(pair[7 + 3*i + 1])
                target_y = float(pair[7 + 3*i + 2])
                pairList.append([ID, sourceIdx, current_x, current_y, targetIdx, target_x, target_y])

    pairDF = pd.DataFrame(pairList)

    return pairDF

def plot_camera(ax, C, R, scale=1.0, color='r'):
    """
    Plots a 3D camera object as a pyramid (frustum approximation).
    
    Args:
        ax: Matplotlib 3D axis.
        C: Camera center (3D point).
        R: Rotation matrix (3x3).
        scale: Scaling factor for the camera size.
        color: Color of the camera object.
    """
    # Define the base of the pyramid (camera frustum) in the camera's local frame
    base = np.array([
        [-1, -1, 2],  # Bottom-left
        [1, -1, 2],   # Bottom-right
        [1, 1, 2],    # Top-right
        [-1, 1, 2]    # Top-left
    ]) * scale

    apex = np.array([0, 0, 0])  # Apex of the pyramid (camera center in local frame)
    # print(C)
    # Transform the base and apex to the world frame
    base_world = (R @ base.T).T + C
    apex_world = C

    # Create the faces of the pyramid (triangles)
    faces = [
        [apex_world, base_world[0], base_world[1]],  # Front face
        [apex_world, base_world[1], base_world[2]],  # Right face
        [apex_world, base_world[2], base_world[3]],  # Back face
        [apex_world, base_world[3], base_world[0]],  # Left face
        base_world  # Bottom face (quadrilateral)
    ]

    # Plot the pyramid
    ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=0.5))

def PlotCameraPts(Cset, Rset, Xset, save_dir, filename):
    """
    Plots all 3D points and camera objects in the X-Z plane with fixed axis limits.
    
    Args:
        Cset: List of camera centers.
        Rset: List of rotation matrices.
        Xset: List of 3D points (one set for each camera pose).
        save_dir: Directory to save the plot.
        filename: Name of the output plot file.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'g', 'blue', 'black']  # Different colors for each pose

    for i in range(len(Cset)):
        # Plot 3D points for the current camera pose
        X = np.array(Xset[i])[:, 1:4]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors[i], marker='o',s=1, label=f'3D Points (Pose {i+1})')

        # Plot the camera object
        plot_camera(ax, Cset[i].flatten(), Rset[i], scale=1,color=colors[i])

    # Set fixed viewing angle to the X-Z plane
    ax.view_init(elev=0, azim=90)

    # Set axis limits
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-15, 15])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points and Camera Objects (X-Z Plane)')
    ax.legend()

    # Save and display the plot
    plt.savefig(f"{save_dir}/{filename}")
    plt.show()

def PlotPtsCams(Cset, Rset, Xset, save_dir, filename, show_pos=True):
    """
    Plots the selected 3D points and the camera pose.
    
    Args:
        Cset: List of camera centers (1 camera pose selected).
        Rset: List of rotation matrices (1 rotation matrix selected).
        Xset: List of triangulated 3D points.
        save_dir: Output directory for saving the plot.
        filename: Filename for the saved plot.
        show_pos: Whether to display the camera position or not.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X=np.squeeze(np.array(Xset))[:,1:4]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', marker='o',s=1, label='3D Points')

        # Plot the camera object
    plot_camera(ax, Cset[0].flatten(), Rset[0], scale=1,color='blue')

    # Set fixed viewing angle to the X-Z plane
    ax.view_init(elev=0, azim=90)

    # Set axis limits
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-15, 15])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points and Camera Objects (X-Z Plane)')
    ax.legend()

    # Save and display the plot
    plt.savefig(f"{save_dir}/{filename}")
    plt.show()
