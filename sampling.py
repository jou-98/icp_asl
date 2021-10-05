import open3d as o3d 
import numpy as np 
from time import time
import glob
from sklearn.cluster import KMeans
from pathlib import Path
import copy
import random as rdm
from numpy.linalg import norm as pnorm
from numpy.random import default_rng


# Vanilla sampling method
def downsample(arr1,arr2,method='default',n_fps=0.3):
    pts1 = np.copy(arr1)
    pts2 = np.copy(arr2)
    max_points = max(pts1.shape[0],pts2.shape[0])
    min_points = min(pts1.shape[0],pts2.shape[0])

    if method == 'default':
        pts1 = pts1[:min_points,:] if pts1.shape[0] >= pts2.shape[0] else pts1
        pts2 = pts2[:min_points,:] if pts2.shape[0] > pts1.shape[0] else pts2
        return pts1,pts2
    if method == 'np_rand':
        rng = default_rng()
        idx = rng.choice(max_points,size=min_points,replace=False)
    elif method == 'py_rand':
        idx = rdm.sample(range(max_points),min_points)
    elif method == 'grid':
        pass 
    elif method == 'fps':
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts1)
        pcd2.points = o3d.utility.Vector3dVector(pts2)
        n_points = min_points*n_fps
        pts1 = fps(pcd1,num_points=n_points)
        pts2 = fps(pcd2,num_points=n_points)
        max_points = max(pts1.shape[0],pts2.shape[0])
        min_points = min(pts1.shape[0],pts2.shape[0])
        idx = rdm.sample(range(max_points),min_points)
    if pts1.shape[0] == max_points: pts1 = pts1[idx,:]
    if pts2.shape[0] == max_points: pts2 = pts2[idx,:]
    return pts1, pts2


def kmeans_pc(pts,n=1000):
    kmeans = KMeans(n_clusters=1000).fit(pts)
    return kmeans.cluster_centers_

def fps(pcd,num_points=2000,return_pts=True):
    pts = np.asarray(pcd.points)
    n = int(pts.shape[0]/num_points)
    pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, n)
    if return_pts: return np.asarray(pcd_new.points) 
    return pcd_new

def grid(pts,grid_size=1e-3):
    pass

def iss(pcd):
    pass
