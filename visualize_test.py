from utils.visualize import *
import numpy as np
import torch
pc_n = np.load('ShapeNetCore.v2.PC15k/02818832/train/5cdbef59581ba6b89a87002a4eeaf610.npy')
tr_idxs = np.arange(2048)
pc = torch.from_numpy(pc_n[tr_idxs,:])
pc = pc.repeat(25, 1, 1)
print(pc.shape)
visualize_pointcloud_batch('test.png' ,
                            pc, None, None,
                            None)