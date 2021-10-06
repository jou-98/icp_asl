import numpy as np
from glob import glob
from utils import *
from metadata import *

confdir = 'stairs/data/icpList.csv'

gt = np.loadtxt(confdir,delimiter=',',skiprows=1,usecols=range(1,17),dtype=np.float32)
gt = gt.reshape((-1,4,4))

def inverse_mat(mat):
    newmat = np.zeros_like(mat)
    newmat[:3,:3] = mat[:3,:3].T
    newmat[:3,3] = -np.matmul(mat[:3,:3].T,mat[:3,3])
    newmat[3,3] = 1.0
    #print(f'Euler angle for old matrix is {euler_angles(mat)}, for new matrix is {euler_angles(newmat)}')
    return newmat

which = 'stairs'

files = glob(which+'/data/Hokuyo_*.csv')
data_size = len(files)

pc = np.random.rand(4,3)
#print(pc)
#pcd1 = open_csv(get_fname(0,suffix,which), astype='pcd')
for i in range(data_size):
    for j in range(data_size):
        gti = gt[i]
        gtj = gt[j]
        print(f'This should be identity:\n{np.rint(np.matmul(inverse_mat(gti),gti)).astype(np.uint16)}')
        #agg_r = np.matmul(gti[:3,:3],gtj[:3,:3].T)
        #agg_t = np.matmul(gti[:3,3],gtj[:3,3])
        agg = np.matmul(gtj,inverse_mat(gti))
        #print(agg)