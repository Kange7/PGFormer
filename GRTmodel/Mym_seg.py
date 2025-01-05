import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

# from .model_utils import *
from .pointnet_util import *

import torch.nn.functional as F
import numpy as np
from models.blocks import *


    
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


# 2. -> PosE for Local Geometry Extraction  
class Pos_Enc(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        self.lpe = nn.Parameter(torch.zeros(1))
   
    def forward(self, xyz, x):
        # B, _, G, K = xyz.shape
        B, G, K = xyz.shape
        # feat_dim = self.out_dim // (self.in_dim * 2)
        feat_dim = self.out_dim // 2

        feat_range = torch.arange(feat_dim).float().cuda()   
 
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)

        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)  # div
        

        sin_embed = torch.sin(div_embed) 
        cos_embed = torch.cos(div_embed)

        position_embed = torch.cat([sin_embed, cos_embed], -1)  

        position_embed = position_embed.permute(0, 3, 1, 2).contiguous()
        position_embed = position_embed.view(B, self.out_dim, G, K)
    
        # Weigh
        position_embed = position_embed + self.lpe   
        position_embed = position_embed * self.lpe

        return position_embed


# 2. -> Transformer 
class TransformerBlock(nn.Module):
    def __init__(self, channel, hidden_d, group_heads) -> None:   
        super().__init__()
        self.fc1 = nn.Linear(channel, hidden_d)  
        self.fc2 = nn.Linear(hidden_d, channel)
        self.PE = Pos_Enc(3, hidden_d, 1000, 100)  
        self.fc_delta = nn.Sequential(
            nn.Linear(3, hidden_d),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(hidden_d, hidden_d)
        )

        self.fc_gamma = nn.Sequential(
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(hidden_d, hidden_d)
        )
        
        self.num = group_heads

        self.fc_gamma_2 = nn.Sequential(
            nn.Linear(hidden_d//self.num, hidden_d//self.num),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(hidden_d//self.num, hidden_d//self.num)
        )

        self.w_qs = nn.Linear(hidden_d, hidden_d, bias=False)  
        self.w_ks = nn.Linear(hidden_d, hidden_d, bias=False)
        self.w_vs = nn.Linear(hidden_d, hidden_d, bias=False)

        self.w_qs_2 = nn.Linear(hidden_d//self.num, hidden_d//self.num , bias=False)
        self.w_ks_2 = nn.Linear(hidden_d//self.num, hidden_d//self.num , bias=False)
        

    # xyz: b x n x 3, features: b x n x f
    def forward(self, features):   #  knn_xyz, knn_x

        xyz = features[:, :3]  
        xyz = xyz.unsqueeze(0).permute(0, 2, 1)
        features = features.unsqueeze(0)
        
        pre = features

        x = self.fc1(features)  

        # The size of each group
        group_size1 = x.size(-1) // self.num
        group_size2 = x.size(-2) // self.num


        # Initialize the group list
        attn1 = []
        attn2 = []

        # group
        for i in range(self.num):
            # print(i)
            group_feature1 = x[:, :, i*group_size1:(i+1)*group_size1]  # [24, 320, 64]
            
            q = self.w_qs_2(group_feature1)     

            k = self.w_ks_2(group_feature1)  

            attn11 = self.fc_gamma_2 (q - k ) 
      
            attn11 = F.softmax(attn11 / np.sqrt(k.size(-1)), dim=-2) 
            attn1.append(attn11) 
        attn1 = torch.cat(attn1, dim=-1) # [24, 320, 256]
        
        
        for j in range(self.num):
            
            group_feature2 = x[:, i*group_size2:(i+1)*group_size2, :]  # [24, 320, 64]
            
            q = self.w_qs(group_feature2)     
            k = self.w_ks(group_feature2)
            
            attn22 = self.fc_gamma (q - k ) 
            attn22 = F.softmax(attn22 / np.sqrt(k.size(-1)), dim=-2)
            attn2.append(attn22) 
        attn2 = torch.cat(attn2, dim=-2)


        # attn = attn1 + attn2
        # The shape of the tensor
        shape1 = attn1.shape
        shape2 = attn2.shape


        # Identify the dimensions that need to be padded
        if shape1[-2] > shape2[-2]:
            
            padding_size = shape1[-2] - shape2[-2]
           
            attn2_padded = torch.nn.functional.pad(attn2, (0, 0, 0, padding_size), mode='constant', value=0)
            attn = attn1 + attn2_padded
        elif shape1[-2] < shape2[-2]:
            
            padding_size = shape2[-2] - shape1[-2]
            
            attn1_padded = torch.nn.functional.pad(attn1, (0, 0, 0, padding_size), mode='constant', value=0)
            attn = attn1_padded + attn2
        else:
            attn = attn1 + attn2

        v = self.w_vs(x)

        pos_enc = self.PE(xyz, x)  # b x n x k x f  
        pos_enc = pos_enc.permute(0, 3, 1, 2)
        pos_enc = pos_enc.narrow(-1, 0, 1)
        pos_enc = torch.squeeze(pos_enc, dim=-1) 

        res = torch.einsum('bmf,bmf->bmf', attn, v + pos_enc)

        res = self.fc2(res) + pre
        res = res.squeeze(0) 

        return res


def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)
