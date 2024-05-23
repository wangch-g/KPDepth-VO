import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.pointnet import PointNet
from networks.pointnet_modules import CorrespondenceModule
from networks.depthnet import DepthNet


class KeypointDepthVO(nn.Module):
    def __init__(self, config):
        super(KeypointDepthVO, self).__init__()
        self.config = config

        self.pointnet = PointNet()
        self.depthnet = DepthNet(depth_scale=self.config.num_scales, num_layers=18)

        if self.config.mode == "keypoint":
            self.correspondence_module = CorrespondenceModule()

    def top_k_keypoints(self, scores, coords, descs, ratio=0.3):
        B, N = scores.shape
        k = int(N * ratio)
        scores, indices = torch.topk(scores, k, dim=1)
        coords_ = []
        descs_ = []
        for b in range(B):
            coords_.append(coords[b, :, :][indices[b, :]])
            descs_.append(descs[b, :, :][indices[b, :]])

        return scores, torch.stack(coords_, dim=0), torch.stack(descs_, dim=0) 

    def disp2depth(self, disp, min_depth=0.1, max_depth=100.0):
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth  

    def pointnet_infer(self, img, top_points_ratio):
        score, coord, desc = self.pointnet(img, is_train=False)

        score = score.flatten(start_dim=2).squeeze(1)
        coord = coord.flatten(start_dim=2).transpose(1, 2)
        desc = desc.flatten(start_dim=2).transpose(1, 2)

        topk_score, topk_coord, topk_desc = self.top_k_keypoints(score,
                                                                 coord,
                                                                 desc,
                                                                 ratio=top_points_ratio)

        return topk_score, topk_coord, topk_desc

    def depthnet_infer(self, img):
        disp_list, photosigma_list_cur, depthsigma_list_cur = self.depthnet(img)
        scaled_disp, depth = self.disp2depth(disp_list[0])
        return scaled_disp, depth, photosigma_list_cur[0], depthsigma_list_cur[0]
    
    def inference(self, img, top_points_ratio):

        _, coord, desc = self.pointnet_infer(img, top_points_ratio)
        disp, depth, photosigma, depthsigma = self.depthnet_infer(img) 

        pred = {}
        pred['coord'] = coord
        pred['desc'] = desc
        pred['disp'] = disp
        pred['depth'] = depth
        pred['photosigma'] = photosigma
        pred['depthsigma'] = depthsigma
        
        return pred