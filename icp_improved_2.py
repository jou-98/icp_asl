import numpy as np 
import open3d as o3d 
from utils import * 
from metadata import *
import argparse
from sampling import downsample
from sklearn.neighbors import NearestNeighbors as NN
from scipy.optimize import linear_sum_assignment as LSA
from scipy.spatial.distance import directed_hausdorff as Hausdorff
from scipy.spatial.distance import cosine
from scipy.spatial.distance import correlation
import random as rdm
from Logger import Logger
from copy import deepcopy as dcopy
from time import time
from glob import glob
#import pyflann
from numpy.linalg import norm as pnorm

def my_dist(x,y):
    return correlation(x,y)

def iss(pcd,salient_radius=0.005,non_max_radius=0.005,gamma_21=0.5,gamma_32=0.5):
    return o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                        salient_radius=salient_radius,
                                                        non_max_radius=non_max_radius,
                                                        gamma_21=gamma_21,
                                                        gamma_32=gamma_21)

def voxel_down(pcd1,pcd2,voxel_size):
    pcd_down1 = pcd1.voxel_down_sample(voxel_size)
    pcd_down2 = pcd2.voxel_down_sample(voxel_size)

    key1 = iss(pcd_down1,salient_radius=voxel_size,non_max_radius=voxel_size)
    key2 = iss(pcd_down2,salient_radius=voxel_size,non_max_radius=voxel_size)

    pts1 = np.asarray(key1.points)
    pts2 = np.asarray(key2.points)
    min_points = min(pts1.shape[0],pts2.shape[0])
    if pts1.shape[0] >= pts2.shape[0]:
        idx = rdm.sample(range(pts1.shape[0]),min_points)
        key1.points = o3d.utility.Vector3dVector(pts1[idx])
    else:
        idx = rdm.sample(range(pts2.shape[0]),min_points)
        key2.points = o3d.utility.Vector3dVector(pts2[idx])
    return key1,key2, pcd_down1, pcd_down2


def preprocess_point_cloud(pcd,voxel_size):
    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd,pcd_fpfh

def prepare_dataset(source,target,voxel_size,trans_init=None):
    #print(":: Load two point clouds and disturb initial pose.")
    if trans_init is None: 
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    source_key, target_key, source_down, target_down = voxel_down(source,target,voxel_size)
    #o3d_save_pc(source,'original.ply',pcd_format=True)
    #o3d_save_pc(source_down,'downsampled.ply',pcd_format=True)
    source_key, source_fpfh = preprocess_point_cloud(source_key,voxel_size)
    target_key, target_fpfh = preprocess_point_cloud(target_key,voxel_size)
    return source_down, target_down, source_key, target_key, source_fpfh, target_fpfh

def compute_dist(pts1,pts2,t,idx1,idx2):
    """
    pcd1 = o3d_instance(pts1)
    pcd2 = o3d_instance(pts2)
    pcd1.transform(t)
    dists = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    """
    pts1 = pts_transform(pts1,t)
    #dists = pnorm(pts1[idx1]-pts2[idx2],ord=2,axis=1)
    d1,_,_ = Hausdorff(pts1,pts2)
    d2,_,_ = Hausdorff(pts2,pts1)
    return max(d1,d2)


def correspondence_search(x,y,nbrs=None):
    if nbrs is None: 
        nbrs = NN(n_neighbors=1, algorithm='kd_tree').fit(y)
    dist, idx = nbrs.kneighbors(x)
    return dist.ravel(), idx.ravel(), nbrs

def compute_matrix(x,y):
    dim = x.shape[1]
    # Normalize the points
    centroid_x = np.mean(x, axis=0)
    centroid_y = np.mean(y, axis=0)
    xx = x - centroid_x 
    yy = y - centroid_y
    #print(f'shape of xx is {xx.shape}, shape of yy is {yy.shape}')
    H = np.matmul(xx.T,yy)
    U, S, V = np.linalg.svd(H)
    # last dimension
    if(np.linalg.det(H) < 0):
        V[-1,:] *= -1
    # Rotation matrix
    R = np.matmul(V.T, U.T)
    # Translation matrix
    t = -np.matmul(R,centroid_x.T) + centroid_y.T
    # Homogeneous metrix
    T = np.identity(dim+1)
    T[:dim,:dim] = R 
    T[:dim,dim] = t
    return T, R, t

def compute_init(x,y,idx1=None,idx2=None,disturb=None):
    dim = x.shape[1]

    src = np.ones((dim+1,x.shape[0]))
    dst = np.ones((dim+1,y.shape[0]))
    src[:dim,:] = np.copy(x.T)
    dst[:dim,:] = np.copy(y.T)
    T, R, t = compute_matrix(src[:dim,idx1].T, dst[:dim,idx2].T)
    if disturb is not None: T_agg = np.matmul(T,disturb)
    return T, T_agg 

def icp(x, y, max_iter=100, threshold=0.00001,init=None,logger=None):    
    dim = x.shape[1]
    if init is not None: x = pts_transform(x,init)
    src = np.ones((dim+1,x.shape[0]))
    dst = np.ones((dim+1,y.shape[0]))
    src[:dim,:] = np.copy(x.T)
    dst[:dim,:] = np.copy(y.T)

    d_trend = []
    i = 0
    dist = 0
    T = 0
    prev_error = 0

    time_search = 0
    count_search = 0
    time_matrix = 0
    count_matrix = 0
    metric = 0
    nbrs = None
    while True:
        i += 1
        start = time()
        if nbrs is None: dist, idx, nbrs = correspondence_search(src[:dim,:].T, dst[:dim,:].T)
        if nbrs is not None: dist, idx, _ = correspondence_search(src[:dim,:].T, dst[:dim,:].T, nbrs)
        time_search += time() - start
        count_search += 1
        start = time()
        T, R, t = compute_matrix(src[:dim,:].T, dst[:dim,idx].T)
        time_matrix += time() - start
        count_matrix += 1
        src = np.dot(T, src)
        error = np.mean(dist)
        d_trend.append(error)
        if i % 10 == 0:
            pass
        if np.abs(error-prev_error) <= threshold or i == max_iter:
            metric = cRMS(src[:dim].T, dst[:dim,idx].T)
            break 
        prev_error = error
    start = time()
    T, R, t = compute_matrix(x, src[:dim,:].T)
    time_matrix += time() - start 
    logger.record_nn(time_search)
    logger.record_mat(time_matrix)


    print(f'Time taken to find corresponding points: {round(time_search,3)}s')
    print(f'Average time for each correspondence search {round(time_search/count_search,3)}s')
    print(f'Time taken to find transformation matrix: {round(time_matrix,3)}s')
    print(f'Average time for each matrix computation {round(time_matrix/count_matrix,3)}s')
    print(f'Total time taken for ICP: {round(time_search+time_matrix,3)}')
    print(f'cRMS = {round(metric,3)}')
    return T, dist, i, logger

def calc_one_pair(file1,file2,voxel_size,radius,which='stairs',logger=None,init_thres=0.0001):
    meta_start = time()
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    suffix=''
    f1,f2=file1,file2
    pcd1 = open_csv(get_fname(file1,suffix,which), astype='pcd')
    pcd2 = open_csv(get_fname(file2,suffix,which), astype='pcd')
    print(f'==================== Testing {f1}.ply against {f2}.ply ====================')

    source,target,source_down,target_down,source_fpfh,target_fpfh = prepare_dataset(dcopy(pcd1),dcopy(pcd2),voxel_size)
    arr1 = source_fpfh.data.T
    arr2 = target_fpfh.data.T
    #arr1 = np.asarray(source_down.points)
    #arr2 = np.asarray(target_down.points)
    print(f'Shape of feature matrix is {arr1.shape} and {arr2.shape}.')
    
    """
    pyflann.set_distance_type("minkowski",order=2)
    flann = pyflann.FLANN()
    params = flann.build_index(arr2,algorithm="kmeans",branching=32,iterations=10,target_precision=0.9)
    k_nearest = 300
    result, dists = flann.nn_index(arr1,k_nearest,checks=params["checks"])
    row_ind,col_ind = LSA(dists)
    valid = row_ind
    idx = result[row_ind,col_ind]
    print(f'Sum of costs is {dists[valid, col_ind].sum()}')
    print(f'shape of idx is {idx.shape}, shape of valid is {valid.shape}')
    pts1 = np.asarray(source_down.points)
    pts2 = np.asarray(target_down.points)
    trans_init = compute_init(pts1,pts2,idx1=valid,idx2=idx)
    """
    start = time()
    nbrs = NN(radius=radius,p=2,metric='minkowski',algorithm='kd_tree').fit(arr2)
    count = 0
    mat = np.zeros((arr1.shape[0],arr2.shape[0]))
    valid = []
    for i in range(arr1.shape[0]):
        dist,idx = nbrs.radius_neighbors(arr1[i].reshape(1,-1))
        if idx[0].shape[0] > 0: count += 1; valid.append(i);# print(f'Found {idx[0].shape[0]} neighbours.')
        #if i % 50 == 0: print(f'{i} points computed')
        for j in range(idx.shape[0]):
            mat[i,idx[j]] = dist[j]
    print(f'NN search costs {time()-start}s')
    start = time()
    mat[mat==0] = 1000
    row_ind, col_ind = LSA(mat)
    print(f'Linear sum assignment costs {time()-start}s')
    print(f'Sum of costs is {mat[valid, col_ind[valid]].sum()}')
    print(f'{count} points have as least one neighbour in target point cloud.')
    idx = col_ind[valid]
    pts1 = np.asarray(source_down.points)
    pts2 = np.asarray(target_down.points)
    icp_init, align_init = compute_init(pts1,pts2,idx1=valid,idx2=idx,disturb=trans_init)
    

    dist = compute_dist(pts1,pts2,trans_init,valid,idx)
    pts1 = np.asarray(source_down.points)
    pts2 = np.asarray(target_down.points)
    print(f'Shape of pts1 is {pts1.shape} and pts2 is {pts2.shape}.')
    print(f'Mean chamfer distance after initial transformation is {np.mean(dist)}')
    if np.mean(dist) <= init_thres:
        i = 0
        T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    else:
        T,dist,i,logger = icp(pts1,pts2,max_iter=100,threshold=0.00001,init=icp_init,logger=logger)

    logger.record_meta(time()-meta_start)
    print(f'Done in {time()-meta_start}s.')
    print(f'Done in {i} iterations')
    confdir = get_conf_dir(which=which)
    gt = np.loadtxt(confdir,delimiter=',',skiprows=1,usecols=range(1,17),dtype=np.float32)
    gt = gt.reshape((-1,4,4))
    t1 = gt[file1]
    t2 = gt[file2]
    t1_inv = inverse_mat(t1)
    t2_inv = inverse_mat(t2)
    reGT = rotation_error(t1_inv, t2_inv)
    print(f'Rotation angle between source and target is {round(reGT,3)}')
    # print(f'Extra RE = {rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2)}')

    t_rel = np.matmul(T, align_init)

    rel = np.matmul(t2_inv,t1)
    #rel = mul_transform(t1,t2)
    TE = pnorm(T[:3,3]-rel[:3,3], ord=2)
    RE = rotation_error(T,rel) #pnorm(t1[:3,:3]-t2[:3,:3], ord=2)
    print(f'RE = {round(RE,3)}, TE = {round(TE,3)}')
    #print(f'Extra RE = {rotation_error()}')
    logger.record_re(RE)
    logger.record_reGT(reGT)
    logger.record_te(TE)
    #draw_registration_result(pcd1, pcd2, , filename=file1+'_'+file2+'ex.ply')
    
    if RE >= 15:
        print(f'Computed ground truth transformation is\n{rel}\nCalculated transformation is\n{T}')
        draw_registration_result(pcd1, pcd2, transformation=None, filename=which+'/ours/'+str(file1)+'_'+str(file2)+'_orig.ply')
        draw_registration_result(pcd1, pcd2, t_rel, filename=which+'/ours/'+str(file1)+'_'+str(file2)+'.ply')
        print(f'pcd1 is yellow and pcd2 is blue')
    
    print(f'============================== End of evaluation ==============================\n\n')
    logger.increment()
    return logger


if __name__=='__main__':
    RE_list = []
    TE_list = []
    t_list = []
    logger = Logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stairs', help='dataset to compare')
    parser.add_argument('--radius', type=float, default=0.08, help='radius for NN search')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='size of each voxel')
    parser.add_argument('--overlap', type=float, default=0.6, help='Minimum overlap of pairs of point cloud to be compared')
    FLAGS = parser.parse_args()

    overlap_thresh = FLAGS.overlap
    which = FLAGS.dataset
    voxel_size = FLAGS.voxel_size 
    r = FLAGS.radius
    files = glob(which+'/data/Hokuyo_*.csv')
    data_size = len(files)

    for f1 in range(data_size):
        for f2 in range(data_size):
            if f1 == f2 or not overlap_check(f1,f2,which,overlap_thresh): continue
            print(f'Computing scan_id {f1} against {f2}')
            ger = calc_one_pair(f1,f2,voxel_size=voxel_size,radius=r,which=which,logger=logger)

    print(f'Results for MY OWN algorithm on {which} dataset.')
    print(f'In total, {logger.count} pairs of point clouds are evaluated.')
    print(f'Recall rate is {round(logger.recall(),2)}')
    print(f'Average time to compute each pair is {round(logger.avg_meta(),3)}s')
