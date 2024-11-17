import pandas as pd
import numpy as np
import random
import sys

from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from tqdm import tqdm

def GetInliersRANSAC(x1All, x2All, M=1500, T=0.5):
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
 
    # RANSAC iterations
    for i in tqdm(range(M)):
        # Step 1: Randomly select 8 pairs of points from the source and target images
        random_indices = np.random.choice(len(x1All), 8, replace=False)
        # Extract coordinates without IDs for Fundamental matrix estimation
        x1 = x1All.iloc[random_indices, :]
        x2 = x2All.iloc[random_indices, :]
        # Step 2: Estimate the Fundamental matrix F from the selected 8-point subsets
        # Call EstimateFundamentalMatrix function here.
        F = EstimateFundamentalMatrix(x1, x2)
        # Step 3: Check each point pair to see if it satisfies the epipolar constraint
        inliers = []
        for j in range(len(x1All)):
            # Get homogeneous coordinates for the point pair
            x1 = np.array([x1All.iloc[j,0], x1All.iloc[j,1], 1])
            x2 = np.array([x2All.iloc[j,0], x2All.iloc[j,1], 1])
            
            # Calculate epipolar constraint error: x2^T * F * x1
            error = abs(x2.T @ F @ x1)
            
            # If error is below threshold, add to inliers
            if error < T:
                inliers.append(j)
        
        # Step 4: Update the best Fundamental matrix if the current F has more inliers
        if i == 0 or len(inliers) > len(best_inliers):
            best_inliers = inliers
            FBest = F
            
    # Get the final inlier sets using the best inlier indices
    x1Inlier = x1All.iloc[best_inliers]
    x2Inlier = x2All.iloc[best_inliers]

    return x1Inlier, x2Inlier, FBest
