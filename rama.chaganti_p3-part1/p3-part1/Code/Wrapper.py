"""
This is a startup script to processes a set of images to perform Structure from Motion (SfM) by
extracting feature correspondences, estimating camera poses, and triangulating 3D points, 
performing PnP and Bundle Adjustment.
"""

import numpy as np
import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image


# Import required functions for various steps in the SfM pipeline.

from utils import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from GetInliersRANSAC import GetInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from PnPRANSAC import *
from NonlinearTriangulation import NonlinearTriangulation
from NonlinearPnP import NonlinearPnP
from BuildVisibilityMatrix import *
from BundleAdjustment import BundleAdjustment

################################################################################
# Step 1: Parse all matching files and assign IDs to feature points.
# Each file contains matches between feature points in successive images.
################################################################################

file1 = '../Data/Imgs/matching1.txt'
file2 = '../Data/Imgs/matching2.txt'
file3 = '../Data/Imgs/matching3.txt'
file4 = '../Data/Imgs/matching4.txt'
file5 = '../Data/Imgs/matching5.txt'

"""
Assign Unique IDs to feature points across datasets.
"""

"""
The IndexAllFeaturePoints function takes five text files (file1 through file5), each representing feature point matches from different images. It processes each file to:
1. Extract and clean feature point data.
2. Assign unique identifiers (IDs) to each feature point.
3. Ensure that feature points shared across different files are assigned the same ID.
"""

# Check if processed matching files already exist to avoid redundant processing.
if not os.path.exists('../Data/new_matching1.txt'):
    print("\nProcessing Feature Correspondences from matching files...")
    # Process the matching files to assign a unique ID to each feature point.
    # This enables tracking of each point across multiple images in the dataset.
    # Each feature point across the entire dataset will have a unique index.
    match1DF, match2DF, match3DF, match4DF, match5DF = IndexAllFeaturePoints(file1, file2, file3, file4, file5)

else:
    print('Refined Features Indexes Already Exists')
    # Refer utils.py for color definiton
    print(bcolors.WARNING + "Warning: Continuing with the existing Feature Indexes..." + bcolors.ENDC)


################################################################################
# Step 2: Parse matching file for the first pair of cameras (cameras 1 and 2).
# Each file contains matches between feature points in successive images.
################################################################################

# Define the file path and camera indices for parsing keypoints.
file_path = '../Data/new_matching1.txt'
source_camera_index = 1
target_camera_index = 2

# Execute the keypoint parsing for the specified camera pair.
# The output DataFrame provides a structured set of matched keypoints between two images.
ParseKeypoints_DF = ParseKeypoints(file_path, source_camera_index, target_camera_index)

# Extract coordinates for source and target keypoints
# Select specific columns to represent keypoint coordinates in each image
# - 0: Keypoint ID, which uniquely identifies each match
# - 2, 3: X and Y coordinates of the keypoint in the source image
# - 5, 6: X and Y coordinates of the corresponding keypoint in the target image
source_keypoints = ParseKeypoints_DF[[0, 2, 3]]
target_keypoints = ParseKeypoints_DF[[0, 5, 6]]

# print(f'\nSource Keypoints: "  {source_keypoints[2]}')

# print(f'\nSource Keypoints: "  {target_keypoints[6]}')
source_all_points = source_keypoints[[2, 3]].to_numpy()
target_all_points = target_keypoints[[5, 6]].to_numpy()
################################################################################
# Step 3: RANSAC-based outlier rejection.
# Remove outlier based on fundamental matrix. For every eight points, you can
# get one fundamental matrix. But only one fundamental matrix is correct between
# two images. Optimize and find the best fundamental matrix and remove outliers
# based on the features that corresponds to that fundamental matrix.
# This step refines the matches by removing outliers and retaining inliers.
################################################################################

# Use RANSAC to estimate a robust Fundamental matrix (F) that minimizes the impact of outliers.
# Write a function GetInliersRANSAC that removes outliers and compute Fundamental Matrix
# using initial feature correspondences

source_inliers, target_inliers, F = GetInliersRANSAC(source_keypoints, target_keypoints)
source_inlier_points = source_inliers.to_numpy()
target_inlier_points = target_inliers.to_numpy()
print(F)


# #################################################################################
# # You will write another function 'EstimateFundamentalMatrix' that computes F matrix
# # This function is being called by the 'GetInliersRANSAC' function
# #################################################################################

# # Visualize the final feature correspondences after computing the correct Fundamental Matrix.
# # Write a code to print the final feature matches and compare them with the original ones.

output_path = '../Output/'
DrawMatches('../Data/Imgs/', source_camera_index, target_camera_index, source_all_points, target_all_points, source_inlier_points, target_inlier_points, output_path)

# ################################################################################
# # Step 4: Load intrinsic camera matrix K, which contains focal lengths and 
# # principal point.
# # The calibration file provides intrinsic parameters which are used to calculate
# # the Essential matrix.
# ################################################################################

calib_file = '../Data/Imgs/calibration.txt'
K = process_calib_matrix(calib_file)
print(bcolors.OKCYAN + "\nIntrinsic camera matrix K:" + bcolors.OKCYAN)
print(K, '\n')


# ################################################################################
# # Step 5: Compute Essential Matrix from Fundamental Matrix
# ################################################################################
E = EssentialMatrixFromFundamentalMatrix(F, K)

# ################################################################################
# # Step 6: Extract Camera Poses from Essential Matrix
# # Note: You will obtain a set of 4 translation and rotation from this function
# ################################################################################
Cset, Rset = ExtractCameraPose(E)

# ################################################################################
# # Step 6: Linear Triangulation
# ################################################################################
# # Initialize an empty list to store the 3D points calculated for each camera pose
Xset = []
# Iterate over each camera pose in Cset and Rset
for i in range(4):
    # Perform linear triangulation to estimate the 3D points given:
    # - K: Intrinsic camera matrix
    # - np.zeros((3,1)): The initial camera center (assumes origin for the first camera)
    # - np.eye(3): The initial camera rotation (identity matrix, assuming no rotation for the first camera)
    # - Cset[i]: Camera center for the i-th pose
    # - Rset[i]: Rotation matrix for the i-th pose
    # - x1Inlier: Inlier points in the source image
    # - x2Inlier: Corresponding inlier points in the target image
    Xset_i = LinearTriangulation(K, np.zeros((3,1)), np.eye(3), Cset[i], Rset[i], source_inliers, target_inliers)
    Xset.append(Xset_i)

# ################################################################################
# ## Step 7: Plot all points and camera poses
# # Write a function: PlotPtsCams that visualizes the 3D points and the estimated camera poses.
# ################################################################################

# Arguments:
# - Cset: List of camera centers for each pose.
# - Rset: List of rotation matrices for each pose.
# - Xset: List of triangulated 3D points corresponding to each camera pose.
# - SAVE_DIR: Output directory to save the plot.
# - FourCameraPose.png: Filename for the output plot showing 3D points and camera poses.

PlotCameraPts(Cset, Rset, Xset, output_path, "FourCameraPose.png")




################################################################################
## Step 8: Disambiguate Camera Pose
# Write a function: DisambiguateCameraPose
# DisambiguateCameraPose is called to identify the correct camera pose from multiple
# hypothesized poses. It selects the pose with the most inliers in front of both 
# cameras (i.e., the pose with the most consistent triangulated 3D points).
################################################################################

## Disambiguate camera poses
# Arguments:
# - Cset: List of candidate camera centers for each pose.
# - Rset: List of candidate rotation matrices for each pose.
# - Xset: List of sets of triangulated 3D points for each camera pose.
# Returns:
# - C: The selected camera center after disambiguation.
# - R: The selected rotation matrix after disambiguation.
# - X: The triangulated 3D points for the selected camera pose.
# - selectedIdx: The index of the selected camera pose within the original candidate sets.
C, R, X, selectedIdx = DisambiguateCameraPose(Cset, Rset, Xset)

# Plot the selected camera pose with its 3D points
# This plot shows the selected camera center, orientation, and the corresponding 3D points.
# Arguments:
# - [C]: List containing the selected camera center (wrapping it in a list for compatibility with PlotPtsCams).
# - [R]: List containing the selected rotation matrix.
# - [X]: List containing the 3D points for the selected camera pose.
# - SAVE_DIR: Output directory to save the plot.
# - OneCameraPoseWithPoints.png: Filename for the output plot showing both the camera pose and 3D points.
# - show_pos=True: Enables the display of the camera pose.
PlotPtsCams([C], [R], [X], output_path, "OneCameraPoseWithPoints.png")

################################################################################
## Step 9: Non-Linear Triangulation
# Write a function: NonLinearTriangulation
# Inputs:
# - K: Intrinsic camera matrix of the first camera (3x3).
# - C0, R0: Translation (3x1) and rotation (3x3) of the first camera.
# - Cseti, Rseti: Translations and rotations of other cameras in a list.
# - x1set, x2set: Sets of 2D points in each image for triangulation.
# - X0: Initial 3D points for optimization.
# Output:
# - Returns optimized 3D points after minimizing reprojection error.
# NonlinearTriangulation(K, C0, R0, Cseti, Rseti, x1set, x2set, X0):
################################################################################

# ################################################################################
# # Step 10: PnPRANSAC
# # PnPRANSAC: Function to perform PnP using RANSAC to find the best camera pose 
# # with inliers
# ################################################################################

# ################################################################################
# # Step 11: NonLinearPnP
# # NonLinearPnP: Refines the camera pose (position and orientation) using non-linear 
# # optimization to minimize the reprojection error between observed 2D points and 
# # projected 3D points.
# ################################################################################


# ################################################################################
# # Step 12: BuildVisibilityMatrix
# # BuildVisibilityMatrix: BuildVisibilityMatrix: Constructs a sparse visibility 
# # matrix for bundle adjustment. This matrix indicates which parameters affect 
# # each observation in the optimization process.
# ################################################################################

# ################################################################################
# # Step 13: BundleAdjustment
# # BundleAdjustment: Refines camera poses and 3D point positions to minimize the 
# # reprojection error for a set of cameras and 3D points using non-linear 
# # optimization.
# ################################################################################
