import torch
import torch.nn as nn
import torchvision.transforms as tvf

from networks.pointnet_modules import PointNetEncoder, PointNetDecoder

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        self.encoder = PointNetEncoder()
        self.decoder = PointNetDecoder()
        
    def forward(self, img, is_train=True):
        _, _, H, _ = img.shape
        
        features = self.encoder(img)
        _, _, hc, _ = features[-1].shape

        # If is_train=True, 'desc' is a list, otherwise a tensor [B, 256, H, W].
        score, coord, desc = self.decoder(features, downsample_ratio=int(H/hc), is_train=is_train)
        return score, coord, desc
