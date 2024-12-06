import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from EstimateFundamentalMatrix import EstimateFundamentalMatrix

def GetInliersRANSAC(x1All, x2All, M=1500, T=0.1):
    """
    Estimates the Fundamental matrix using RANSAC and identifies inlier matches
    between two sets of points, rejecting outliers.

    Args:
        x1All (DataFrame): Source image points with IDs and (x, y) coordinates.
        x2All (DataFrame): Target image points with IDs and (x, y) coordinates.
        M (int): Number of RANSAC iterations. Default is 1500.
        T (float): Threshold for inlier selection based on the epipolar constraint. Default is 0.5.

    Returns:
        x1Inlier (DataFrame): Inlier points in the source image.
        x2Inlier (DataFrame): Inlier points in the target image.
        FBest (ndarray): The best estimated Fundamental matrix.
    """
    print("Running RANSAC...")
    # print("Source Keypoints Columns:", x1All.columns)
    # print("Target Keypoints Columns:", x2All.columns)
    
    feature_idex=x1All[[0]].to_numpy()
    x1All = x1All[[2, 3]].to_numpy()
    x2All = x2All[[5, 6]].to_numpy()

    max_inliers = 0
    FBest = None
    x1Inlier = None
    x2Inlier = None

    # RANSAC iterations
    for i in tqdm(range(M)):
        # Step 1: Randomly select 8 pairs of points from the source and target images
        indices = random.sample(range(x1All.shape[0]), 8)
        x1 = x1All[indices]
        x2 = x2All[indices]

        # Step 2: Estimate the Fundamental matrix F from the selected 8-point subsets
        F = EstimateFundamentalMatrix(pd.DataFrame(x1, columns=['x', 'y']), pd.DataFrame(x2, columns=['x', 'y']))

        # Step 3: Check each point pair to see if it satisfies the epipolar constraint
        inliers = []
        for j in range(x1All.shape[0]):
            x1_h = np.array([x1All[j][0], x1All[j][1], 1])  # Homogeneous coordinates
            x2_h = np.array([x2All[j][0], x2All[j][1], 1])  # Homogeneous coordinates
            
            # Calculate the epipolar constraint error for the source-target pair
            error = abs(np.dot(x2_h.T, np.dot(F, x1_h)))
            
            # If the epipolar constraint error is below the threshold T, consider it an inlier
            if error < T:
                inliers.append(j)

        # Step 4: Update the best Fundamental matrix if the current F has more inliers
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            FBest = F
            x1Inlier = x1All[inliers]
            x2Inlier = x2All[inliers]
            feature_idex_inliers=feature_idex[inliers]


    # Convert inliers back to DataFrames
    x1Inlier = [(feature_idex_inliers[i], *x1Inlier[i]) for i in range(len(feature_idex_inliers))]
    x2Inlier = [(feature_idex_inliers[i], *x2Inlier[i]) for i in range(len(feature_idex_inliers))]
    x1Inlier = pd.DataFrame(x1Inlier, columns=['Id','x', 'y'])
    x2Inlier = pd.DataFrame(x2Inlier, columns=['Id','x', 'y'])
    

    return x1Inlier, x2Inlier, FBest