import numpy as np
import open3d as o3d
import copy
from numpy.random import default_rng
import csv
from math import sqrt
from numpy.linalg import norm as pnorm


BIN_DIR = './bin/'
LABELs_DIR = './labels/'


def open_csv(fname, astype='pcd', skiprows=1):
    pts = np.loadtxt(fname,delimiter=',',skiprows=skiprows, dtype=np.float32)
    pts = pts[:,1:4]
    if astype == 'pcd':
        pcd = o3d_instance(pts)
        return pcd
    elif astype == 'pts':
        return pts 
    return pts

def get_conf_dir(name='stairs'):
    if name == 'stairs': return 'stairs/data/icpList.csv'

def get_fname(fname,suffix='./',which='stairs'):
    if which == 'stairs': return suffix+which+'/data/Hokuyo_'+str(fname)+'.csv'
    return None

# Mat1 is the transformation from source to scan_0)
# Mat2 is the transformation from target to scan_0
# returns: 
def mul_transform(mat1, mat2,separate=False):
    rotate = np.dot(mat2[:3,:3],mat1[:3,:3])
    translate = mat1[:3,3] - mat2[:3,3]
    if separate: return rotate, translate
    base = np.zeros((4,4),dtype=np.float32)
    base[:3,:3] = rotate
    base[:3,3] = translate
    base[3,3] = 1.
    print(f'Shape of base is {base.shape}')
    return base


# Computes the coordinate Root-Mean-Squared error between the two
# point clouds.
def cRMS(src, dst):
    diff = src-dst
    if np.max(diff)==np.min(diff): print(f'src is equal to dst!!')
    std = np.std(diff)
    squared = np.apply_along_axis(lambda x:(x[0]**2+x[1]**2+x[2]**2) ,axis=1,arr=diff)
    mean_squared = np.mean(squared)
    #print(f'Max diff is {np.max(diff)} and min diff is {np.min(diff)}')
    #print(f'Standard deviation is {std}')
    return np.sqrt(mean_squared) / (np.max(diff)-np.min(diff))


def render_color(pc,label,ply_path='./00_001000_colored.ply'):
    labels = np.unique(label)
    colors = dict()
    for i in labels:
        color = list(np.random.choice(range(100), size=3)/100)
        colors[i] = color
    rgb = np.zeros((pc.shape[0],3))

    for i in labels:
        label_pos = np.argwhere(np.any(label==i, axis=1))#label[label==i]
        if not i in CLASSES:
            continue
        rgb[label_pos] = colors[i]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(ply_path, pcd) 
    
def pts_transform(pts,t,dim_return=3):
    if pts.shape[1] == 3:
        buf = np.ones((pts.shape[0],pts.shape[1]+1))
        buf[:,:3] = pts
        pts = buf
    pts_new = np.matmul(t,pts.T).T
    if dim_return == 3:
        return pts_new[:,:3]
    else:
        return pts_new

def o3d_read_pc(path,ext='ply'):
    pcd = o3d.io.read_point_cloud(path,format=ext)
    pts = np.asarray(pcd.points)
    return pcd, pts

def o3d_instance(pts,rgb=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None: pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def o3d_save_pc(pts,path,colors=None,pcd_format=False):
    if pcd_format:
        pcd = pts
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None: pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


# Reads a .bin file into numpy array
def read_pc_bin(path='./dataset/sequences/00/velodyne/001000.bin',dim=4):
    scan = np.fromfile(path, dtype=np.float32)
    scan = scan.reshape((-1, dim))     
    return scan[:,:3]

def save_pc(array,ply_path='./00_001000.ply'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array[:,:3])
    o3d.io.write_point_cloud(ply_path, pcd) 


# Reads a .label file into numpy array
def read_label(path='./labels/00_000143.label',dim=2):
    labels = np.fromfile(path, dtype=np.uint16).reshape((-1,dim))
    labels = labels[:,0]
    #print(np.unique(labels,return_counts=True))
    return labels


# Renders a point cloud in black and white, reads either bin file or ply file 
def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

# Draws the result of transformation matrix. Takes two open3d point clouds
# as input, not arrays
def draw_registration_result(source, target, transformation=None, filename='a+b.ply'):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    if transformation is not None: source_temp=source_temp.transform(transformation)

    rgb = np.concatenate((np.asarray(source_temp.colors),np.asarray(target_temp.colors)),axis=0)
    pts = np.concatenate((np.asarray(source_temp.points),np.asarray(target_temp.points)),axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(filename, pcd)

def overlap_check(f1,f2,which='stairs',thresh=0.70):
    if which == 'stairs': fname = './stairs/data/overlap_stairs.csv'
    percentages = np.genfromtxt(fname,delimiter=',',dtype=np.float32)[:,:-1]
    l = len(percentages)
    percentages = percentages.reshape((l,l))
    return percentages[f1,f2] >= thresh 
    



def euler_angles(t):
    x = np.arctan2(t[2,1],t[2,2])
    y = np.arctan2(-t[2,0],np.sqrt(t[2,1]*t[2,1]+t[2,2]*t[2,2]))
    z = np.arctan2(t[1,0],t[0,0])
    return np.rad2deg(np.array([x,y,z]).reshape((-1,1)))


def rotation_error(t1,t2):
    return pnorm(euler_angles(t1) - euler_angles(t2))
    #return pnorm(t1[:3,:3]-t2[:3,:3])

    