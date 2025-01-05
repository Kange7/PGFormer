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


# PosE for Local Geometry Extraction  
class Pos_Enc(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
   
        
    def forward(self, xyz, x):
        # B, _, G, K = xyz.shape
        B, G, K = xyz.shape
        # feat_dim = self.out_dim // (self.in_dim * 2)
        feat_dim = self.out_dim // 2

        feat_range = torch.arange(feat_dim).float().cuda()   
 
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)

        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)  # div除法操作
        # div_embed = torch.div(self.beta * xyz, dim_embed)
        # print(div_embed.size())

        sin_embed = torch.sin(div_embed)  # [24, 3, 256, 85] 这个size已经不对了
        cos_embed = torch.cos(div_embed)

        position_embed = torch.cat([sin_embed, cos_embed], -1)  # [24, 3, 256, 170]  这个size已经不对了
        # print(position_embed.size())

        # position_embed = position_embed.permute(0, 1, 4, 2, 3).contiguous()
        position_embed = position_embed.permute(0, 3, 1, 2).contiguous()
        position_embed = position_embed.view(B, self.out_dim, G, K)
        # position_embed = position_embed.view(B, G, K)

        # Weigh
        # knn_x_w = knn_x + position_embed   # 这里怎么改，以及怎么调用类，还有参数如何传入
        # knn_x_w *= position_embed

        return position_embed

class TransformerBlock(nn.Module):
    def __init__(self, channel, hidden_d, group_heads) -> None:   # 点云维度、隐藏层维度、 K 近邻（貌似邻近的交互不行）
        super().__init__()
        self.fc1 = nn.Linear(channel, hidden_d)  # 这些线性层能换成卷积就换成卷积
        self.fc2 = nn.Linear(hidden_d, channel)
        # self.PE = Pos_Enc(3, hidden_d, 1000, 100)  # 输入输出维度没有说明
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

        self.w_qs = nn.Linear(hidden_d, hidden_d, bias=False)  # 换成卷积
        self.w_ks = nn.Linear(hidden_d, hidden_d, bias=False)
        self.w_vs = nn.Linear(hidden_d, hidden_d, bias=False)

        self.w_qs_2 = nn.Linear(hidden_d//self.num, hidden_d//self.num , bias=False)
        self.w_ks_2 = nn.Linear(hidden_d//self.num, hidden_d//self.num , bias=False)
        

        # self.q_conv = nn.Conv2d(hidden_d, hidden_d , 1)
        # self.k_conv = nn.Conv2d(hidden_d, hidden_d , 1)
        # self.v_conv = nn.Conv2d(hidden_d, hidden_d , 1)

        # self.q_conv_2 = nn.Conv2d(hidden_d//self.num, hidden_d//self.num , 1)
        # self.k_conv_2 = nn.Conv2d(hidden_d//self.num, hidden_d//self.num , 1)
        # self.v_conv_2 = nn.Conv2d(hidden_d//self.num, hidden_d//self.num , 1)

    # xyz: b x n x 3, features: b x n x f
    def forward(self, features, xyz):   # 对应于 knn_xyz, knn_x

        pre = features
        
        x = self.fc1(features.permute(0, 2, 1))   # 以下换成卷积
        # q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        # print(x.size())

        # 确定每个分组的大小
        group_size1 = x.size(-1) // self.num
        group_size2 = x.size(-2) // self.num

        # print(group_size)


        # 初始化分组列表
        attn1 = []
        attn2 = []

        # 分组
        for i in range(self.num):
            # print(i)
            group_feature1 = x[:, :, i*group_size1:(i+1)*group_size1]  # [24, 320, 64]
            
            q = self.w_qs_2(group_feature1)     
            # k = self.w_ks_2(group_feature1.permute(2, 1, 0)).permute(2, 1, 0)  # 这里说明了卷积和线性层使用的维度不一样，所以保留了一个
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


        attn = attn1 + attn2
        v = self.w_vs(x)
        
        dists = square_distance(xyz, xyz)  # 计算距离
        knn_idx = dists.argsort()[:, :, :10]  # b x n x k  排序 选取最近的K个邻居
        knn_xyz = index_points(xyz, knn_idx)  # 获取每个点的K个邻居点的坐标
        # pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)

        # pos_enc = self.PE(xyz, x)  # b x n x k x f  # 位置编码换成三角函数形式
        # pos_enc = pos_enc.permute(0, 3, 1, 2)
        # pos_enc = pos_enc.narrow(-1, 0, 1)
        # pos_enc = torch.squeeze(pos_enc, dim=-1)

        res = torch.einsum('bmf,bmf->bmf', attn, v + pos_enc)
        # res = torch.einsum('bmf,bmf->bmf', attn, v)
        
        res = self.fc2(res).permute(0, 2, 1) + pre

        # Output new features and points
        new_features = res
        new_points = xyz
        
        return new_features, new_points
    

# # 以下两个类组合形成原始点嵌入模块
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


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx




