import open3d as o3d
import numpy as np
import copy
import argparse
from utils import *
from time import time
from sklearn.neighbors import NearestNeighbors as NN
from numpy.random import default_rng
from sampling import *
from Logger import Logger
from matplotlib import pyplot as plt
from metadata import *
from glob import glob


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

def icp(x, y, max_iter=1000, threshold=0.00001, logger=None):    
    dim = x.shape[1]

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
            #print(f'Iteration {i}: error = {error.round(5)}')
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
    print(f'Total time taken: {round(time_search+time_matrix,3)}')
    print(f'cRMS = {round(metric,3)}')
    return T, dist, i, logger

def calc_one_icp(file1, file2, logger=None, which='bunny'):
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    pcd1 = open_csv(get_fname(file1,suffix,which), astype='pcd')
    pcd2 = open_csv(get_fname(file2,suffix,which), astype='pcd')
    source = copy.deepcopy(pcd1)
    target = copy.deepcopy(pcd2)
    
    source.transform(trans_init)

    pts1 = np.asarray(source.points)
    pts2 = np.asarray(target.points)
    start = time()
    pts1, pts2 = downsample(pts1,pts2,method='fps',n_fps=0.3) # Changed from np_rand
    print(f'Shape of pts1 is {pts1.shape}')
    print(f'Shape of pts2 is {pts2.shape}')

    time_sample = time() - start
    print(f'Time taken to downsample the point cloud: {round(time_sample,3)}s')
    logger.record_sampling(time_sample)
    #print(f'Shape of pts1 is {pts1.shape}; Shape of pts2 is {pts2.shape}')
    T, dist, i, logger = icp(pts1, pts2, max_iter=200, threshold=0.00001,logger=logger)
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

    t_rel = np.matmul(T, trans_init)

    rel = np.matmul(t2_inv,t1)
    #rel = mul_transform(t1,t2)
    TE = pnorm(T[:3,3]-rel[:3,3], ord=2)
    RE = rotation_error(t_rel,rel) #pnorm(t1[:3,:3]-t2[:3,:3], ord=2)
    print(f'RE = {round(RE,3)}, TE = {round(TE,3)}')
    #print(f'Extra RE = {rotation_error()}')
    logger.record_re(RE)
    logger.record_reGT(reGT)
    logger.record_te(TE)
    #draw_registration_result(pcd1, pcd2, , filename=file1+'_'+file2+'ex.ply')
    
    if RE >= 15:
        print(f'Computed ground truth transformation is\n{rel}\nCalculated transformation is\n{T}')
        draw_registration_result(pcd1, pcd2, transformation=None, filename=which+'/baseline_vanilla/'+str(file1)+'_'+str(file2)+'_orig.ply')
        draw_registration_result(pcd1, pcd2, t_rel, filename=which+'/baseline_vanilla/'+str(file1)+'_'+str(file2)+'.ply')
        print(f'pcd1 is yellow and pcd2 is blue')
    
    print(f'============================== End of evaluation ==============================\n\n')
    logger.increment()
    return logger

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stairs', help='Subset of the ASL dataset to compare')
    parser.add_argument('--overlap', type=float, default=0.6, help='Minimum overlap of pairs of point cloud to be compared')
    FLAGS = parser.parse_args()

    overlap_thresh = FLAGS.overlap

    RE_list = []
    TE_list = []
    t_list = []
    log = Logger()
    
    which = FLAGS.dataset
    files = glob(which+'/data/Hokuyo_*.csv')
    data_size = len(files)

    for f1 in range(data_size):
        for f2 in range(data_size):
            if f1 == f2 or not overlap_check(f1,f2,which,overlap_thresh): continue
            print(f'Computing scan_id {f1} against {f2}')
            log = calc_one_icp(f1,f2,log,which=which)

        
    print(f'Results for (mostly) unmodified ICP algorithm on {which} dataset.')
    print(f'In total, {log.count} pairs of point clouds are evaluated.')
    print(f'Recall rate is {round(log.recall(re_thres=re_thres,te_thres=te_thres),2)}')
    print(f'Average time to compute each pair is {round(log.avg_all(),3)}s')
    print(f'Average time to downsample the point cloud is {round(log.avg_sampling(),3)}')
    print(f'Average time to find correspondence is {round(log.avg_nn(),3)}s')
    print(f'Average time to compute transformation matrix is {round(log.avg_mat(),3)}s')

