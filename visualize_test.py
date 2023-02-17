from utils.visualize import *
import numpy as np
import torch
pc_n = np.load('ShapeNetCore.v2.PC15k/04379243/train/1a1fb603583ce36fc3bd24f986301745.npy')
tr_idxs = np.arange(2048)
pc = torch.from_numpy(pc_n[tr_idxs,:])
pc = pc.repeat(25, 1, 1)
print(pc.shape)
visualize_pointcloud_batch('test.png' ,
                            pc, None, None,
                            None)