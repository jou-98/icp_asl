import numpy as np 
from scipy.optimize import linear_sum_assignment as LSA
from utils import *
from sklearn.neighbors import NearestNeighbors as NN
from time import time
from sampling import *
from numpy.linalg import norm as pnorm
from copy import deepcopy
from Logger import Logger
import argparse
from glob import glob



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

def compute_init(x,y,idx1=None,idx2=None,trans_init=None):
    dim = x.shape[1]

    src = np.ones((dim+1,x.shape[0]))
    dst = np.ones((dim+1,y.shape[0]))
    src[:dim,:] = np.copy(x.T)
    dst[:dim,:] = np.copy(y.T)
    T, R, t = compute_matrix(src[:dim,idx1].T, dst[:dim,idx2].T)
    newT = T
    if trans_init is not None: newT = np.matmul(T,trans_init)
    return T, newT


def calc_one_pair(file1,file2,which='stairs',trans_init=None,voxel_size=0.5,radius=0.5,thres=0.1,logger=None):
    print(f'\n==================== Testing Hokuyo_{file1} against Hokuyo_{file2} ====================')

    if trans_init is None: trans_init = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    #print(f'Rotation angle of initial pose disturbance is {euler_angles(trans_init).reshape((-1))}')



    pcd1 = open_csv(f'{which}/data/Hokuyo_{file1}.csv')
    pcd2 = open_csv(f'{which}/data/Hokuyo_{file2}.csv')

    meta_start = time()
    ckpt = time()
    # Downsample
    source = deepcopy(pcd1)
    source.transform(trans_init)
    target = deepcopy(pcd2)
    down1, down2 = grid_pcd(source,target,voxel_size)

    #print(f'Downsampling takes {time()-ckpt}s.')
    ckpt = time()

    # Estimate normal 
    radius_normal = voxel_size * 2
    down1.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    down2.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Compute FPFH features
    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(
        down1,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(
        down2,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    #print(f'FPFH computation takes {time()-ckpt}s.')


    fpts1 = fpfh1.data.T
    fpts2 = fpfh2.data.T
    print(f'Shape of fpts1 is {fpts1.shape} and shape of fpts2 is {fpts2.shape}')

    nbrs = NN(radius=radius,p=2,metric='minkowski',algorithm='kd_tree').fit(fpts2)

    idx = nbrs.kneighbors(fpts1,1,False).reshape((-1))



    ckpt = time() 

    pts1 = np.array(down1.points)
    pts2 = np.array(down2.points)
    #print(f'Max of pts1 is {np.max(pts1,axis=0)}, Min of pts1 is {np.min(pts1,axis=0)}')
    # Affinity matrix
    aff = np.zeros((fpts1.shape[0],fpts1.shape[0]))

    target = pts2.take(idx,axis=0)
    #print(f'Shape of target is {target.shape}, shape of idx is {idx.shape} and shape of pts2 is {pts2.shape}')

    for i in range(aff.shape[0]):
        # Precompute the distance between source points
        si_curr = pts1[i]
        snorm = pnorm(si_curr-pts1,axis=1)
        # Precompute the distance between corresponding target points
        ti_curr = target[i]
        tnorm = pnorm(ti_curr-target,axis=1)

        for j in range(aff.shape[1]):
            if i > j: continue
            if i==j: aff[i][j] = 100.0; continue
            dist = abs(snorm[j] - tnorm[j])/snorm[j]
            aff[i][j] = dist 
            aff[j][i] = dist
    #print(f'Max of aff is {np.max(aff)} and min of aff is {np.min(aff)}')
    #aff = np.exp(-0.3*aff)

    #print(f'Computing affinity matrix takes {time()-ckpt}s.')
    ckpt = time() 

    bestset = []

    for i in range(pts1.shape[0]):
        currset = [i,]
        if np.all(aff[i]>thres): print(f'Skipping {i}'); continue
        for j in range(pts1.shape[0]):
            diff = aff[j][currset]
            if np.all(diff<=thres): currset.append(j)
        if len(currset) > len(bestset): bestset = currset

    print(f'Size of best set is {len(bestset)}')

    #print(f'Computing the best set takes {time()-ckpt}s.')
    ckpt = time()

    # Draw two pcds
    draw_registration_result(o3d_instance(pts1[bestset]), o3d_instance(pts2[idx[bestset]]), None, filename='test.ply')

    icp_init, trans_init = compute_init(pts1,pts2,idx1=bestset,idx2=idx[bestset],trans_init=trans_init)
    #print(f'Initial transformation computation takes {time()-ckpt}s.')


    dist = compute_dist(pts1,pts2,trans_init,valid,idx)
    down1, down2 = second_down(pcd1,pcd2,voxel_size)
    pts1 = np.asarray(down1.points)
    pts2 = np.asarray(down2.points)
    print(f'Size of pts1 is {len(pts1)}, size of pts2 is {len(pts2)}')
    print(f'Mean chamfer distance after initial transformation is {np.mean(dist)}')
    if np.mean(dist) <= init_thres:
        i = 0
        T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    else:
        T,dist,i,logger = icp(pts1,pts2,max_iter=100,threshold=0.00001,init=trans_init,logger=logger)



    logger.record_meta(time()-meta_start)

    confdir = get_conf_dir(which=which)
    gt = np.loadtxt(confdir,delimiter=',',skiprows=1,usecols=range(1,17),dtype=np.float32)
    gt = gt.reshape((-1,4,4))
    t1 = gt[file1]
    t2 = gt[file2]
    t1_inv = inverse_mat(t1)
    t2_inv = inverse_mat(t2)
    reGT = rotation_error(t1_inv, t2_inv)
    print(f'Rotation angle between source and target is {round(reGT,3)}')

    t_rel = np.matmul(T, trans_init) # trans_init

    rel = np.matmul(t2_inv,t1)
    #rel = mul_transform(t1,t2)
    TE = pnorm(t_rel[:3,3]-rel[:3,3], ord=2)
    RE = rotation_error(t_rel,rel) #pnorm(t1[:3,:3]-t2[:3,:3], ord=2)
    print(f'RE = {round(RE,3)}, TE = {round(TE,3)}')

    logger.record_re(RE)
    logger.record_reGT(reGT)
    logger.record_te(TE)

    if RE >= 15 and RE <= 30: draw_registration_result(pcd1, pcd2, t_rel, filename=which+'/ours/'+str(file1)+'_'+str(file2)+'.ply')
    logger.increment()
    print(f'Executing gm.py takes {time()-meta_start}s.')
    return logger



if __name__=='__main__':
    RE_list = []
    TE_list = []
    t_list = []
    logger = Logger()

    trans_init = np.asarray([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stairs', help='dataset to compare')
    parser.add_argument('--radius', type=float, default=0.5, help='radius for NN search')
    parser.add_argument('--thres', type=float, default=0.1, help='Threshold for best set selection')
    parser.add_argument('--voxel_size', type=float, default=0.5, help='size of each voxel')
    parser.add_argument('--overlap', type=float, default=0.6, help='Minimum overlap of pairs of point cloud to be compared')
    FLAGS = parser.parse_args()

    overlap_thresh = FLAGS.overlap
    which = FLAGS.dataset
    voxel_size = FLAGS.voxel_size 
    r = FLAGS.radius
    thres = FLAGS.thres
    files = glob(which+'/data/Hokuyo_*.csv')
    data_size = len(files)

    for f1 in range(data_size):
        for f2 in range(data_size):
            if f1 == f2 or not overlap_check(f1,f2,which,overlap_thresh): continue
            logger = calc_one_pair(f1,f2,which=which,trans_init=trans_init,voxel_size=voxel_size,radius=r,thres=thres,logger=logger)
    
    print(f'Results for MY OWN algorithm on {which} dataset.')
    print(f'In total, {logger.count} pairs of point clouds are evaluated.')
    print(f'Recall rate is {round(logger.recall(),2)}')
    print(f'Average time to compute each pair is {round(logger.avg_meta(),3)}s')
    