import numpy as np
from glob import glob
from utils import *
from metadata import *

confdir = 'stairs/data/icpList.csv'

gt = np.loadtxt(confdir,delimiter=',',skiprows=1,usecols=range(1,17),dtype=np.float32)
gt = gt.reshape((-1,4,4))

which = 'stairs'

files = glob(which+'/data/Hokuyo_*.csv')
data_size = len(files)

pc = np.random.rand(4,3)
print(pc)
gt0 = gt[0,:3,:3]
#pcd1 = open_csv(get_fname(0,suffix,which), astype='pcd')
for i in range(data_size):
    #pcd2 = open_csv(get_fname(i,suffix,which), astype='pcd')
    gt_33 = gt[i,:3,:3]
    gt_tl = gt[i,:3,3]
    agg = np.matmul()
    #agg = np.matmul(gt_33,gt_33.T)
    #print(f'Rotation error between 0 and {i} is \n{rotation_error(gt0,gt_33)}')
    #result = np.matmul(pc,agg)
    #print(f'Original pc is\n{pc}\nAfter transformation, pc is\n{result}')
    #draw_registration_result(pcd2, pcd1, transformation=gt[i], filename=f'0_{i}.ply')
    #print(f'Ground truth matrix multiplied by its inverse is\n{np.matmul(gt_33,gt_33.T)}')
    #print(f'The matrix is\n{gt_33}\nand the inverse of its inverse is {np.linalg.inv(np.linalg.inv(gt_33))}')