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
from tqdm import tqdm


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
#print(source_inliers['Id'])
source_inlier_points = source_inliers.to_numpy()
target_inlier_points = target_inliers.to_numpy()

camera_indices_source = pd.DataFrame({'camera_indices': [1] * len(source_inlier_points)})
camera_indices_target = pd.DataFrame({'camera_indices': [2] * len(target_inlier_points)})
source_with_camera=pd.concat([camera_indices_source,source_inliers],axis=1)
target_with_camera=pd.concat([camera_indices_target,target_inliers],axis=1)
inlier_points_2d=pd.concat([source_with_camera,target_with_camera],ignore_index=True)

#print(inlier_points_2d)
print(bcolors.OKCYAN + "\nFundamental Matrix F:" + bcolors.OKCYAN)
print(F, '\n')


# Visualize the final feature correspondences after computing the correct Fundamental Matrix.
output_path = '../Output/'
DrawMatches('../Data/Imgs/', source_camera_index, target_camera_index, source_all_points, target_all_points, source_inlier_points, target_inlier_points, output_path)

################################################################################
# Step 4: Load intrinsic camera matrix K, which contains focal lengths and 
# principal point.
# The calibration file provides intrinsic parameters which are used to calculate
# the Essential matrix.
################################################################################

calib_file = '../Data/Imgs/calibration.txt'
K = process_calib_matrix(calib_file)
print(bcolors.BOLD + "\nIntrinsic camera matrix K:" + bcolors.BOLD)
print(K, '\n')


################################################################################
# Step 5: Compute Essential Matrix from Fundamental Matrix
################################################################################
E = EssentialMatrixFromFundamentalMatrix(F, K)
print(bcolors.OKCYAN + "\nEssential Matrix E:"  + bcolors.OKCYAN)
print(E, '\n')

################################################################################
# Step 6: Extract Camera Poses from Essential Matrix
# Note: You will obtain a set of 4 translation and rotation from this function
################################################################################
Cset, Rset = ExtractCameraPose(E)

################################################################################
# Step 6: Linear Triangulation
################################################################################
# Initialize an empty list to store the 3D points calculated for each camera pose
Xset = []
# Iterate over each camera pose in Cset and Rset
for i in range(4):
    # Perform linear triangulation to estimate the 3D points
    Xset_i = LinearTriangulation(K, np.zeros((3,1)), np.eye(3), Cset[i], Rset[i], source_inliers, target_inliers)
    Xset.append(Xset_i)

################################################################################
## Step 7: Plot all points and camera poses
################################################################################

PlotCameraPts(Cset, Rset, Xset, output_path, "FourCameraPose.png")

################################################################################
## Step 8: Disambiguate Camera Pose
################################################################################

C, R, X, selectedIdx = DisambiguateCameraPose(Cset, Rset, Xset)

# Plot the selected camera pose with its 3D points
PlotPtsCams([C], [R], [X], output_path, "OneCameraPoseWithPoints.png")

################################################################################
## Step 9: Non-Linear Triangulation
################################################################################
print("\nPerforming Non-Linear Triangulation...")
points_3D= NonlinearTriangulation(K, np.zeros((3, 1)), np.eye(3), C, R, source_inliers, target_inliers, X)
# print("\nOptimized 3D Points:")
#print(X_refined)
PlotPtsCams([C,C], [R,R], [X,points_3D], output_path, "Refined3DPoints.png")
# xset=target_inliers[['x','y']].to_numpy()
# ################################################################################
# # Step 10: PnP RANSAC
# ################################################################################
c_g_set=[np.zeros((3, 1))]
R_g_set=[np.eye(3)]
c_g_set.append(C.reshape(3,1))
R_g_set.append(R)
#print(c_g_set,R_g_set)
for i in tqdm(range(2,6)):
            source_camera_index = i
            target_camera_index = i+1
            file_path=f'../Data/new_matching{i-1}.txt'
            # print(file_path)
            ParseKeypoints_DF = ParseKeypoints(file_path, source_camera_index, target_camera_index)
        
            source_keypoints = ParseKeypoints_DF[[0, 2, 3]]
            target_keypoints = ParseKeypoints_DF[[0, 5, 6]]
            source_inliers, target_inliers, F = GetInliersRANSAC(source_keypoints, target_keypoints,display=False)
            camera_indices_source = pd.DataFrame({'camera_indices': [i-1] * len(source_inliers)})
            camera_indices_target = pd.DataFrame({'camera_indices': [i+1] * len(target_inliers)})
            # print(camera_indices_source)
            source_with_camera=pd.concat([camera_indices_source,source_inliers],axis=1)
            target_with_camera=pd.concat([camera_indices_target,target_inliers],axis=1)
            inlier_points_2d=pd.concat([inlier_points_2d,source_with_camera],ignore_index=True)
            inlier_points_2d=pd.concat([inlier_points_2d,target_with_camera],ignore_index=True)
            #print(inlier_points_2d)

            xset=target_inliers.to_numpy()
            
            col1 = np.array([x[0][0] if isinstance(x[0], np.ndarray) else x[0] for x in xset])
            col2 = np.array([x[0] for x in points_3D]) 
            common_values = np.intersect1d(col1, col2)
            X_nrefined = np.array([row for row in points_3D if row[0] in common_values])
            xnset = np.array([[
                row[0][0] if isinstance(row[0], np.ndarray) else row[0], 
                row[1], 
                row[2]
            ] for row in xset if (row[0][0] if isinstance(row[0], np.ndarray) else row[0]) in common_values])

            Cnew, Rnew, best_inliers_X, best_inliers_x=PnPRANSAC(X_nrefined, xnset, K, M=2000, T=30, display=False)
            Cnew, Rnew = NonlinearPnP(best_inliers_X, best_inliers_x, K, Cnew, Rnew)
            c_g_set.append(Cnew)
            R_g_set.append(Rnew)
            file_path=f'../Data/new_matching{i}.txt'

            ParseKeypoints_DF = ParseKeypoints(file_path, source_camera_index, target_camera_index)

            source_keypoints = ParseKeypoints_DF[[0, 2, 3]]
            target_keypoints = ParseKeypoints_DF[[0, 5, 6]]
            source_inliers, target_inliers, F = GetInliersRANSAC(source_keypoints, target_keypoints,display=False)
            camera_indices_source = pd.DataFrame({'camera_indices': [i] * len(source_inliers)})
            camera_indices_target = pd.DataFrame({'camera_indices': [i+1] * len(target_inliers)})

            
            source_with_camera=pd.concat([camera_indices_source,source_inliers],axis=1)
            target_with_camera=pd.concat([camera_indices_target,target_inliers],axis=1)
            inlier_points_2d=pd.concat([inlier_points_2d,source_with_camera],ignore_index=True)
            inlier_points_2d=pd.concat([inlier_points_2d,target_with_camera],ignore_index=True)


            #print(inlier_points_2d)
            X_new = LinearTriangulation(K, c_g_set[i-1], R_g_set[i-1], Cnew, Rnew, source_inliers, target_inliers)
            

            X_new = NonlinearTriangulation(K, c_g_set[i-1], R_g_set[i-1], Cnew, Rnew, source_inliers, target_inliers, X_new)
            points_3D=np.concatenate((points_3D,X_new))
            

            # print(c_g_set)
            n_cameras = len(c_g_set) 

            n_points = points_3D.shape[0]  
            camera_indices = []
            point_indices = []
            xall = [] 


            point_map = {int(point_id): idx for idx, point_id in enumerate(points_3D[:, 0])}



            point_ids = inlier_points_2d['Id'].values
            x_coords = inlier_points_2d[['x','y']].values 
            camera_id= inlier_points_2d['camera_indices'].values

            for pid, coords,id in zip(point_ids, x_coords,camera_id):
                if int(pid[0]) in point_map:
                    camera_indices.append(id)
                    point_indices.append(point_map[int(pid[0])])
                    xall.append(coords)

  
            # for i in range(2, 6):
            #     file_path = f'../Data/new_matching{i-1}.txt'
            #     ParseKeypoints_DF = ParseKeypoints(file_path, i, i+1)
                

            #     point_ids = ParseKeypoints_DF[0].values
            #     x_coords = ParseKeypoints_DF[[2, 3]].values  
                

            #     for pid, coords in zip(point_ids, x_coords):
            #         if int(pid) in point_map:
            #             camera_indices.append(i-2)
            #             point_indices.append(point_map[int(pid)])
            #             xall.append(coords)


            camera_indices = np.array(camera_indices)
            point_indices = np.array(point_indices)
            xall = np.array(xall)


           

            try:
                CoptAll, RoptAll, XoptAll = BundleAdjustment(c_g_set, R_g_set, points_3D, K,
                                                            n_cameras, n_points, camera_indices, point_indices, xall)
            except ValueError as e:
                print(f"\nError in Bundle Adjustment:")
                print(f"Error message: {str(e)}")
                raise
            # print(CoptAll,RoptAll,XoptAll)
            c_g_set=CoptAll
            R_g_set=RoptAll
            points_3D=XoptAll
            
            

# print(c_g_set)
# print(R_g_set)


c_g_set=np.array(c_g_set)
R_g_set=np.array(R_g_set)
PlotfinalCams(c_g_set, R_g_set, [points_3D], output_path, "afterallCameraProjection_2D.png")
################################################################################
# Step 11: NonLinearPnP
# NonLinearPnP: Refines the camera pose (position and orientation) using non-linear 
# optimization to minimize the reprojection error between observed 2D points and 
# projected 3D points.
################################################################################
print("\nPerforming Non-Linear PnP...")
Copt, Ropt = NonlinearPnP(best_inliers_X, best_inliers_x, K, Cnew, Rnew)

################################################################################
# Step 12: BuildVisibilityMatrix
# BuildVisibilityMatrix: BuildVisibilityMatrix: Constructs a sparse visibility 
# matrix for bundle adjustment. This matrix indicates which parameters affect 
# each observation in the optimization process.
################################################################################


################################################################################
# Step 13: BundleAdjustment
# BundleAdjustment: Refines camera poses and 3D point positions to minimize the 
# reprojection error for a set of cameras and 3D points using non-linear 
# optimization.
################################################################################
