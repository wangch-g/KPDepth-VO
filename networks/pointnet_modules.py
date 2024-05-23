import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math

from einops.einops import rearrange

from torchvision import models
from utils import image_grid

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=256, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        # self.scale = 2 * math.pi
        self.scale = math.pi

    def forward(self, coord):
        y_embed = coord[:, 1, :]
        x_embed = coord[:, 0, :]

        # normlized (x, y)
        y_embed = y_embed * self.scale
        x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats//2, dtype=torch.float32, device=coord.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(0, 2, 1)

        return pos

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        return self.body(x.transpose(1, 2)).transpose(1, 2)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DilationConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilationConv3x3, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        
        self.maxpool2x2 = nn.MaxPool2d(2, 2)

    def forward(self, x):

        self.features = []
        x = (x - 0.45) / 0.225
        x = self.conv1(x)
        x = self.maxpool2x2(x)
        x = self.conv2(x)
        self.features.append(x)
        x = self.maxpool2x2(x)
        x = self.conv3(x)
        self.features.append(x)
        x = self.maxpool2x2(x)
        x = self.conv4(x)
        self.features.append(x)

        return self.features
    
class PointNetDecoder(nn.Module):
    def __init__(self):
        super(PointNetDecoder, self).__init__()

        # score head
        self.score_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # score out
        self.score_out = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        
        # location head
        self.loc_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # location out
        self.loc_out = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.shift_out = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        
        # descriptor out
        self.des_out_0 = DilationConv3x3(64, 256)
        self.des_out_1 = DilationConv3x3(128, 256)
        
    def forward(self, input_features, downsample_ratio=8, is_train=True):
        x0 = input_features[0]
        x1 = input_features[1]
        x2 = input_features[2]

        B, _, hc, wc = x2.shape
        
        # score head
        score_x = self.score_head(x2)
        score_x = self.score_out(score_x)
        score = score_x.sigmoid()
        
        border_mask = torch.ones(B, hc, wc)
        border_mask[:, 0] = 0
        border_mask[:, hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        score = score * border_mask.to(score.device)
        
        # location head
        coord_x = self.loc_head(x2)        
        coord_cell = self.loc_out(coord_x).tanh()
        
        shift_ratio = self.shift_out(coord_x).sigmoid() * 2.0

        step = (downsample_ratio-1) / 2.
        center_base = image_grid(B, hc, wc,
                                 dtype=coord_cell.dtype,
                                 device=coord_cell.device,
                                 ones=False, normalized=False).mul(downsample_ratio) + step

        coord_un = center_base.add(coord_cell.mul(shift_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=downsample_ratio*wc-1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=downsample_ratio*hc-1)

        # descriptor block
        desc_block = []
        desc_block.append(self.des_out_0(x0))
        desc_block.append(self.des_out_1(x1))

        if not is_train:
            coord_norm = coord[:, :2].clone()
            coord_norm[:, 0] = (coord_norm[:, 0] / (float(downsample_ratio*wc-1)/2.)) - 1.
            coord_norm[:, 1] = (coord_norm[:, 1] / (float(downsample_ratio*hc-1)/2.)) - 1.
            coord_norm = coord_norm.permute(0, 2, 3, 1)

            desc0 = torch.nn.functional.grid_sample(desc_block[0], coord_norm)         
            desc1 = torch.nn.functional.grid_sample(desc_block[1], coord_norm)
            
            desc = desc0 + desc1
            desc = desc.div(torch.unsqueeze(torch.norm(desc, p=2, dim=1), 1))  # Divide by norm to normalize.

            return score, coord, desc
        return score, coord, desc_block

class CorrespondenceModule(nn.Module):
    def __init__(self, match_type='dual_softmax'):
        super(CorrespondenceModule, self).__init__()
        self.match_type = match_type

        if self.match_type == 'dual_softmax':
            self.temperature = 0.1
        else:
            raise NotImplementedError()
 
    def forward(self, source_desc, target_desc):
        b, c, h, w = source_desc.size()       
     
        source_desc = source_desc.div(torch.unsqueeze(torch.norm(source_desc, p=2, dim=1), 1)).view(b, -1, h*w)
        target_desc = target_desc.div(torch.unsqueeze(torch.norm(target_desc, p=2, dim=1), 1)).view(b, -1, h*w)

        if self.match_type == 'dual_softmax':
            sim_mat = torch.einsum("bcm, bcn -> bmn", source_desc, target_desc) / self.temperature
            confidence_matrix = F.softmax(sim_mat, 1) * F.softmax(sim_mat, 2)
        else:
            raise NotImplementedError()
        
        return confidence_matrix