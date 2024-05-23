'''
This code was ported from existing repos
[LINK] https://github.com/nianticlabs/monodepth2
'''
import torch
import torch.nn as nn

from networks.depthnet_modules import DepthNetEncoder, DepthNetDecoder

class DepthNet(nn.Module):
    def __init__(self, depth_scale, num_layers=18):
        super(DepthNet, self).__init__()
        self.depth_scale = depth_scale
        self.encoder = DepthNetEncoder(num_layers=num_layers, pretrained=True)
        self.decoder = DepthNetDecoder(self.encoder.num_ch_enc, scales=range(depth_scale))

    def forward(self, img):
        features = self.encoder(img)
        outputs = self.decoder(features)
        disp_list = []
        photosigma_list = []
        depthsigma_list = []
        for i in range(self.depth_scale):
            disp = outputs['disp', i]
            disp_list.append(disp)
            photosigma = outputs['photosigma', i]
            photosigma_list.append(photosigma)
            depthsigma = outputs['depthsigma', i]
            depthsigma_list.append(depthsigma)

        return disp_list, photosigma_list, depthsigma_list