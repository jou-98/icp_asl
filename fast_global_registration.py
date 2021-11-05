import open3d as o3d 
import numpy as np
from utils import *
from time import time
from glob import glob
import argparse
from Logger import Logger 
from numpy.linalg import norm as pnorm
from metadata import *
from copy import deepcopy as dcopy


def prepare_dataset(source,target,voxel_size,trans_init=None):
    print(":: Load two point clouds and disturb initial pose.")
    if trans_init is None:
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def calc_one_fgr(file1,file2,voxel_size=0.001,which='bunny',logger=None):    
    meta_start = time()
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    suffix=''
    f1,f2=file1,file2
    pcd1 = open_csv(get_fname(file1,suffix,which), astype='pcd')
    pcd2 = open_csv(get_fname(file2,suffix,which), astype='pcd')
    print(f'==================== Testing {f1}.ply against {f2}.ply ====================')

    
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(dcopy(pcd1),dcopy(pcd2),voxel_size,trans_init)
    start = time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time() - start))
    print(f'Size of source is {len(np.array(source_down.points))}, size of target is {len(np.array(target_down.points))}')
    logger.record_meta(time()-meta_start)
    T = result_fast.transformation 
    t_rel = np.matmul(T, trans_init)
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

    rel = np.matmul(t2_inv,t1)
    #rel = mul_transform(t1,t2)
    TE = pnorm(t_rel[:3,3]-rel[:3,3], ord=2)
    RE = rotation_error(t_rel,rel) #pnorm(t1[:3,:3]-t2[:3,:3], ord=2)
    print(f'RE = {round(RE,3)}, TE = {round(TE,3)}')
    #print(f'Extra RE = {rotation_error()}')
    logger.record_re(RE)
    logger.record_reGT(reGT)
    logger.record_te(TE)
    logger.increment()
    print(f'Recall so far is {round(logger.recall(),2)}')
    print(f'============================== End of evaluation ==============================\n\n')
    return logger


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='stairs', help='dataset to compare')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='size of each voxel')
    parser.add_argument('--overlap', type=float, default=0.6, help='Minimum overlap of pairs of point cloud to be compared')
    FLAGS = parser.parse_args()

    voxel_size = FLAGS.voxel_size
    which = FLAGS.dataset
    overlap_thresh = FLAGS.overlap

    files = glob(which+'/data/Hokuyo_*.csv')
    data_size = len(files)

    RE_list = []
    TE_list = []
    t_list = []
    log = Logger()

    for f1 in range(data_size):
        for f2 in range(data_size):
            if f1 == f2 or not overlap_check(f1,f2,which,overlap_thresh): continue
            print(f'Computing scan_id {f1} against {f2}')
            log = calc_one_fgr(f1,f2,voxel_size=voxel_size,which=which,logger=log)
    
    print(f'Results for fast global registration algorithm on {which} dataset.')
    print(f'In total, {log.count} pairs of point clouds are evaluated.')
    print(f'Recall rate is {round(log.recall(),2)}')
    print(f'Average time to compute each pair is {round(log.avg_meta(),3)}s')
    #print(f'Average time to downsample the point cloud is {round(log.avg_sampling(),3)}')
    #print(f'Average time to find correspondence is {round(log.avg_nn(),3)}s')
    #print(f'Average time to compute transformation matrix is {round(log.avg_mat(),3)}s')



