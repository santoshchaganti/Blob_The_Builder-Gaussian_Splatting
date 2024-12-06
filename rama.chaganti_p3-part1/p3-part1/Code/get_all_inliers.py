import numpy as np
from utils import *
from GetInliersRANSAC import GetInliersRANSAC

file_path = '../Data/new_matching1.txt'
source_camera_index = 1
target_camera_index = 2


ParseKeypoints_DF = ParseKeypoints(file_path, source_camera_index, target_camera_index)

source_keypoints = ParseKeypoints_DF[[0, 2, 3]]
target_keypoints = ParseKeypoints_DF[[0, 5, 6]]

source_all_points = source_keypoints[[2, 3]].to_numpy()
target_all_points = target_keypoints[[5, 6]].to_numpy()

source_inliers, target_inliers, F = GetInliersRANSAC(source_keypoints, target_keypoints)
print(f"source_inliers: ", source_inliers)


for file in file_path:
    
    for i in range(6):
        for j in range(6):
            source_camera_index = i
            target_camera_index = j+1
            ParseKeypoints_DF = ParseKeypoints(file_path, source_camera_index, target_camera_index)
            source_keypoints = ParseKeypoints_DF[[0, 2, 3]]
            target_keypoints = ParseKeypoints_DF[[0, 5, 6]]
            source_all_points = source_keypoints[[2, 3]].to_numpy()
            target_all_points = target_keypoints[[5, 6]].to_numpy()
            source_inliers, target_inliers, F = GetInliersRANSAC(source_keypoints, target_keypoints)
            source_inliers 
            

            
