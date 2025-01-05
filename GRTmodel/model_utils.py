import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
import numpy as np

from .pointnet_util import index_points, square_distance, knn_point, farthest_point_sample


# FPS + k-NN    
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        # FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x
    

# Linear layer 1
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)
    

class SE_Block(nn.Module):
    def __init__(self,ch_in,reduction=16):
        super(SE_Block,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in,ch_in//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in//reduction,ch_in,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1)
        return x*y.expand_as(x)
