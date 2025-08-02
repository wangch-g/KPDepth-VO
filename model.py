import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.pointnet import PointNet
from networks.pointnet_modules import CorrespondenceModule, DifferenceAttentionModule
from networks.depthnet import DepthNet


class KeypointDepthVO(nn.Module):
    def __init__(self, config):
        super(KeypointDepthVO, self).__init__()
        self.config = config

        self.pointnet = PointNet()
        self.depthnet = DepthNet(depth_scale=self.config.num_scales, num_layers=18)
        self.point_filter = DifferenceAttentionModule(dim=256, num_heads=4)

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

    def score_filter(self, filter, scores, coords, descs, thres=0.5):
        mask = filter.gt(thres)

        scores = scores[mask]
        coords = coords[mask]
        descs = descs[mask]

        return scores, coords, descs
    
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
    
    def inference_v1(self, img, top_points_ratio):

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
    
    def inference_full(self, img, top_points_ratio, cls_score=0.5):
        _, _, H, W = img.shape
        
        topk_score, topk_coord, topk_desc = self.pointnet_infer(img, top_points_ratio)

        # Normalize the coordinates from ([0, h], [0, w]) to ([-1, 1], [-1, 1]).
        topk_coord_norm = topk_coord.clone()
        topk_coord_norm = topk_coord_norm # [B, N, 2]
        topk_coord_norm[:, :, 0] = (topk_coord_norm[:, :, 0] / (float(W - 1) / 2.)) - 1.
        topk_coord_norm[:, :, 1] = (topk_coord_norm[:, :, 1] / (float(H - 1) / 2.)) - 1.

        top_filter = self.point_filter(topk_coord_norm.transpose(1, 2), topk_desc.transpose(1, 2))
        _, coord, desc = self.score_filter(top_filter, topk_score, topk_coord, topk_desc, thres=cls_score)

        disp, depth, photosigma, depthsigma = self.depthnet_infer(img)

        pred = {}
        pred['coord'] = coord
        pred['desc'] = desc
        pred['disp'] = disp
        pred['depth'] = depth
        pred['photosigma'] = photosigma
        pred['depthsigma'] = depthsigma
        
        return pred
