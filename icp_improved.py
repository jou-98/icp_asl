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
from time import time
import pyflann
from numpy.linalg import norm as pnorm

def my_dist(x,y):
    return correlation(x,y)


def voxel_down(pcd1,pcd2,voxel_size):
    pcd_down1 = pcd1.voxel_down_sample(voxel_size)
    pcd_down2 = pcd2.voxel_down_sample(voxel_size)
    pts1 = np.asarray(pcd_down1.points)
    pts2 = np.asarray(pcd_down2.points)
    min_points = min(pts1.shape[0],pts2.shape[0])
    if pts1.shape[0] >= pts2.shape[0]:
        idx = rdm.sample(range(pts1.shape[0]),min_points)
        pcd_down1.points = o3d.utility.Vector3dVector(pts1[idx])
    else:
        idx = rdm.sample(range(pts2.shape[0]),min_points)
        pcd_down2.points = o3d.utility.Vector3dVector(pts2[idx])
    return pcd_down1,pcd_down2


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


def prepare_dataset(src,dst,voxel_size):
    #print(":: Load two point clouds and disturb initial pose.")
    source,_ = o3d_read_pc(src)
    target,_ = o3d_read_pc(dst)
    source_down, target_down = voxel_down(source,target,voxel_size)
    o3d_save_pc(source,'original.ply',pcd_format=True)
    o3d_save_pc(source_down,'downsampled.ply',pcd_format=True)
    source_down, source_fpfh = preprocess_point_cloud(source_down,voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_down,voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

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

def compute_init(x,y,idx1=None,idx2=None):
    dim = x.shape[1]

    src = np.ones((dim+1,x.shape[0]))
    dst = np.ones((dim+1,y.shape[0]))
    src[:dim,:] = np.copy(x.T)
    dst[:dim,:] = np.copy(y.T)
    T, R, t = compute_matrix(src[:dim,idx1].T, dst[:dim,idx2].T)
    return T

def icp(x, y, max_iter=100, threshold=0.00001,init=None,log=None):    
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
    log.record_nn(time_search)
    log.record_mat(time_matrix)


    print(f'Time taken to find corresponding points: {round(time_search,3)}s')
    print(f'Average time for each correspondence search {round(time_search/count_search,3)}s')
    print(f'Time taken to find transformation matrix: {round(time_matrix,3)}s')
    print(f'Average time for each matrix computation {round(time_matrix/count_matrix,3)}s')
    print(f'Total time taken for ICP: {round(time_search+time_matrix,3)}')
    print(f'cRMS = {round(metric,3)}')
    return T, dist, i, log

def calc_one_pair(file1,file2,voxel_size,radius,which='bunny',log=None,init_thres=0.0001):
    meta_start = time()
    suffix=''
    f1,f2=file1,file2
    if which == 'bunny':
        suffix = 'bunny/data/'
        if 'bun' in file1: f1 = file1[-3:]
        if 'bun' in file2: f2 = file2[-3:]
    elif which == 'happy_stand':
        suffix = 'happy_stand/data/'
        f1 = file1[16:]
        f2 = file2[16:]
    elif which == 'happy_side':
        suffix = 'happy_side/data/'
        f1 = file1[15:]
        f2 = file2[15:]
    elif which == 'happy_back':
        suffix = 'happy_back/data/'
        f1 = file1[15:]
        f2 = file2[15:]
    print(f'==================== Testing {f1}.ply against {f2}.ply ====================')

    pcd1,pcd2,source_down,target_down,source_fpfh,target_fpfh = prepare_dataset(suffix+file1+'.ply',suffix+file2+'.ply',voxel_size)
    #arr1 = source_fpfh.data.T
    #arr2 = target_fpfh.data.T
    arr1 = np.asarray(source_down.points)
    arr2 = np.asarray(target_down.points)
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
    trans_init = compute_init(pts1,pts2,idx1=valid,idx2=idx)
    

    dist = compute_dist(pts1,pts2,trans_init,valid,idx)
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    print(f'Mean chamfer distance after initial transformation is {np.mean(dist)}')
    if np.mean(dist) <= init_thres:
        i = 0
        T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    else:
        T,dist,i,log = icp(pts1,pts2,max_iter=100,threshold=0.00001,init=trans_init,log=log)

    log.record_meta(time()-meta_start)
    print(f'Done in {i} iterations')
    print(f'Done in {time()-meta_start}s.')
    confdir = get_conf_dir(name=which)
    tx1,ty1,tz1,qx1,qy1,qz1,qw1 = get_quat(confdir,file1)
    tx2,ty2,tz2,qx2,qy2,qz2,qw2 = get_quat(confdir,file2)
    t1 = quat_to_mat(tx1,ty1,tz1,qx1,qy1,qz1,qw1)
    t2 = quat_to_mat(tx2,ty2,tz2,qx2,qy2,qz2,qw2)
    reGT = rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2)
    print(f'Rotation angle between source and target is {round(reGT,3)}')
    mat = np.matmul(trans_init[:3,:3],t1[:3,:3])

    qx1, qy1, qz1, qw1 = mat_to_quat(mat)
    TE = pnorm(t1[:3,3]-(t2[:3,3]-trans_init[:3,3]), ord=2)
    RE = rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2)
    print(f'For initial estimation, RE = {round(RE,3)}, TE = {round(TE,3)}')
    draw_registration_result(pcd1, pcd2, trans_init, filename=which+'/baseline_ours/'+f1+'_'+f2+'_init.ply')


    mat = np.matmul(T[:3,:3],mat)
    qx1, qy1, qz1, qw1 = mat_to_quat(mat)
    if RE > 15: 
        TE = pnorm(t1[:3,3]-(t2[:3,3]-T[:3,3]-trans_init[:3,3]), ord=2)
        RE = rotation_error(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2)
    print(f'RE = {round(RE,3)}, TE = {round(TE,3)}')
    log.record_re(RE)
    log.record_reGT(reGT)
    log.record_te(TE)

    draw_registration_result(pcd1, pcd2, np.dot(T,trans_init), filename=which+'/baseline_ours/'+f1+'_'+f2+'.ply')
    print(f'============================== End of evaluation ==============================\n\n')
    log.increment()
    return log


if __name__=='__main__':
    RE_list = []
    TE_list = []
    t_list = []
    log = Logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bunny', help='dataset to compare')
    parser.add_argument('--radius', type=float, default=0.08, help='radius for NN search')
    parser.add_argument('--voxel_size', type=float, default=0.002, help='size of each voxel')
    FLAGS = parser.parse_args()

    which = FLAGS.dataset
    voxel_size = FLAGS.voxel_size 
    r = FLAGS.radius

    if which == dataset[0]:
        for i in range(len(bunny_files)):
            for j in range(len(bunny_files)):
                if i != j:
                    log = calc_one_pair(bunny_files[i],bunny_files[j],voxel_size=voxel_size,radius=r,which='bunny',log=log)
    elif which == dataset[1]:
        for i in range(len(stand_files)):
            for j in range(len(stand_files)):
                if np.abs(j-i) < 4 and j != i:
                    log = calc_one_pair(stand_files[i],stand_files[j],voxel_size=voxel_size,radius=r,which='happy_stand',log=log)
    elif which == dataset[2]:
        for i in range(len(side_files)):
            for j in range(len(side_files)):
                if np.abs(j-i) < 4 and j != i:
                    log = calc_one_pair(side_files[i],side_files[j],voxel_size=voxel_size,radius=r,which='happy_side',log=log)
                    break
    elif which == dataset[3]:
        for i in range(len(back_files)):
            for j in range(len(back_files)):
                if np.abs(j-i) < 4 and j != i:
                    log = calc_one_pair(back_files[i],back_files[j],voxel_size=voxel_size,radius=r,which='happy_back',log=log)

    print(f'Results for MY OWN algorithm on {which} dataset.')
    print(f'In total, {log.count} pairs of point clouds are evaluated.')
    print(f'Recall rate is {round(log.recall(),2)}')
    print(f'Average time to compute each pair is {round(log.avg_meta(),3)}s')
