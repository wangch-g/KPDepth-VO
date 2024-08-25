import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation import eval_depth

from model import KeypointDepthVO
import torch
from tqdm import tqdm
import pdb
import cv2
import numpy as np
import yaml

def disp2depth(disp, min_depth=0.001, max_depth=80.0):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def resize_depths(gt_depth_list, pred_disp_list):
    pred_depth_list = []
    pred_disp_resized = []
    for i in range(len(pred_disp_list)):
        h, w = gt_depth_list[i].shape
        pred_disp = cv2.resize(pred_disp_list[i], (w,h))
        pred_depth = 1.0 / (pred_disp + 1e-4)
        pred_depth_list.append(pred_depth)
        pred_disp_resized.append(pred_disp)
    
    return pred_depth_list, pred_disp_resized

def test_eigen_depth(cfg, model):
    print('Evaluate depth using eigen split.')
    filenames = open('./splits/eigen/test_files.txt').readlines()
    pred_disp_list = []
    for i in range(len(filenames)):
        path1, idx, _ = filenames[i].strip().split(' ')
        img = cv2.imread(os.path.join(os.path.join(cfg.raw_data_dir, path1), 'image_02/data/'+str(idx)+'.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, (cfg.kitti_hw[1], cfg.kitti_hw[0]))
        img_input = torch.from_numpy(img_resize / 255.0).float().cuda().unsqueeze(0).permute(0,3,1,2)
        disp, _, photosigma, depthsigma = model.depthnet_infer(img_input)
        disp = disp[0].detach().cpu().numpy()
        photosigma = photosigma.detach().squeeze().cpu()
        depthsigma = depthsigma.detach().squeeze().cpu()
        disp = disp.transpose(1,2,0)
        pred_disp_list.append(disp)
    
    gt_depths = np.load('./splits/eigen/gt_depths.npz', allow_pickle=True)['data']
    pred_depths, _ = resize_depths(gt_depths, pred_disp_list)
    eval_depth_res = eval_depth(gt_depths, pred_depths)
    abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = eval_depth_res
    sys.stderr.write(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10} \n".
        format('abs_rel', 'sq_rel', 'rms', 'log_rms',
                'a1', 'a2', 'a3'))
    sys.stderr.write(
        "{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f} \n".
        format(abs_rel, sq_rel, rms, log_rms, a1, a2, a3))
    
    return eval_depth_res

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description="Testing.")
    arg_parser.add_argument('--gpu', type=int, default=0, help='gpu id.')
    arg_parser.add_argument('--pretrained_model', type=str, default=None, help='directory for loading flow pretrained models')
    arg_parser.add_argument('--raw_data_dir', type=str, default=None, help='')
    arg_parser.add_argument('--kitti_hw', type=int, nargs='+', default=[320, 1024], help='')
    arg_parser.add_argument('--mode', type=str, default='keypoint_depth', help='')
    arg_parser.add_argument('--num_scales', type=int, default=1, help='')
    
    cfg = arg_parser.parse_args()

    model = KeypointDepthVO(cfg)

    torch.cuda.set_device(cfg.gpu)
    model.cuda()
    weights = torch.load(cfg.pretrained_model, map_location='cuda:{}'.format(cfg.gpu))
    model.load_state_dict(weights['model_state'], strict=False)
    model.eval()
    print('Model Loaded.')

    depth_res = test_eigen_depth(cfg, model)

