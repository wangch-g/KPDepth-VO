import os
import yaml
import warnings
import copy
import cv2
import numpy as np
from tqdm import tqdm
import torch

from model import KeypointDepthVO
warnings.filterwarnings("ignore")

def save_traj(path, poses):
    """
    path: file path of saved poses
    poses: list of global poses
    """
    f = open(path, 'w')
    for i in range(len(poses)):
        pose = poses[i].flatten()[:12] # [3x4]
        line = " ".join([str(j) for j in pose])
        f.write(line + '\n')
    print('Trajectory Saved.')

def projection(xy, points, h_max, w_max):
    # Project the triangulation points to depth map.
    # Directly correspondence mapping rather than projection.
    # xy: [N, 2] points: [3, N]
    depth = np.zeros((h_max, w_max))
    xy_int = np.around(xy).astype('int')

    # Ensure all the correspondences are inside the image.
    y_idx = (xy_int[:, 0] >= 0) * (xy_int[:, 0] < w_max)
    x_idx = (xy_int[:, 1] >= 0) * (xy_int[:, 1] < h_max)
    idx = y_idx * x_idx
    xy_int = xy_int[idx]
    points_valid = points[:, idx]

    depth[xy_int[:, 1], xy_int[:, 0]] = points_valid[2]
    return depth

def unprojection(xy, depth, K):
    # xy: [N, 2] image coordinates of match points
    # depth: [N] depth value of match points
    N = xy.shape[0]
    # initialize regular grid
    ones = np.ones((N, 1))
    xy_h = np.concatenate([xy, ones], axis=1)
    xy_h = np.transpose(xy_h, (1,0)) # [3, N]
    #depth = np.transpose(depth, (1,0)) # [1, N]
    
    K_inv = np.linalg.inv(K)
    points = np.matmul(K_inv, xy_h) * depth
    points = np.transpose(points) # [N, 3]
    return points

def cv_triangulation(matches, pose):
    # matches: [N, 4], the correspondence xy coordinates
    # pose: [4, 4], the relative pose trans from 1 to 2
    xy1 = matches[:, :2].transpose()
    xy2 = matches[:, 2:].transpose() # [2, N]
    pose1 = np.eye(4)
    pose2 = pose1 @ pose
    points = cv2.triangulatePoints(pose1[:3], pose2[:3], xy1, xy2)
    points /= points[3]

    points1 = pose1[:3] @ points
    points2 = pose2[:3] @ points
    return points1, points2

class KPDepthVO():
    def __init__(self, config):
        self.img_dir = config.sequences_root
        self.seq_id = config.sequence
        self.new_img_h = config.new_hw[0]
        self.new_img_w = config.new_hw[1]
        self.top_points_ratio = config.top_points_ratio
        self.max_depth = config.max_depth
        self.min_depth = config.min_depth
        self.inlier_min_num = config.inlier_min_num
        self.inlier_min_parallax = config.inlier_min_parallax
        self.pose_ransac_thre = config.pose_ransac_thre
        self.pose_ransac_times = config.pose_ransac_times
        self.PnP_ransac_iter = config.PnP_ransac_iter
        self.PnP_ransac_thre = config.PnP_ransac_thre
        self.PnP_ransac_times = config.PnP_ransac_times

        self.cam_intrinsics = None
    
    def load_kitti(self):
        # load rescaled camera intrinsics
        raw_img_h = 370.0
        raw_img_w = 1226.0
        path = os.path.join(self.img_dir, self.seq_id) + '/calib.txt'
        with open(path, 'r') as f:
            lines = f.readlines()
        data = lines[-1].strip('\n').split(' ')[1:]
        data = [float(k) for k in data]
        data = np.array(data).reshape(3, 4)
        cam_intrinsics = data[:3, :3]
        cam_intrinsics[0, :] = cam_intrinsics[0, :] * self.new_img_w / raw_img_w
        cam_intrinsics[1, :] = cam_intrinsics[1, :] * self.new_img_h / raw_img_h

        self.cam_intrinsics = cam_intrinsics

        # return kitti images
        path = self.img_dir
        seq = self.seq_id
        seq_dir = os.path.join(path, seq)
        image_dir = os.path.join(seq_dir, 'image_2')
        num = len(os.listdir(image_dir))
        images = []
        for i in range(num):
            image = cv2.imread(os.path.join(image_dir, '%.6d'%i)+'.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.new_img_w, self.new_img_h))
            images.append(image)
        return images

    def process_video(self, images, model):
        '''
        Process a sequence to get scale consistent trajectory results.
        '''
        poses = []
        global_pose = np.eye(4)
        # The first one global pose is origin.
        poses.append(copy.deepcopy(global_pose))
        seq_len = len(images)
        # K = self.cam_intrinsics
        # K_inv = np.linalg.inv(self.cam_intrinsics)
        img_ref = images[0]
        pred_ref = self.get_prediction(img_ref, model)

        for i in tqdm(range(1, seq_len)):
            img_cur = images[i]
            pred_cur = self.get_prediction(img_cur, model)

            matched_ref, matched_cur = self.bf_match(pred_ref['coord'], pred_ref['desc'], pred_cur['coord'], pred_cur['desc'])
            rel_pose = np.eye(4)
            pose, inliers_ref, inliers_cur = self.solve_pose(matched_ref, matched_cur)

            rel_pose[:3, :3] = copy.deepcopy(pose[:3, :3])
            if np.linalg.norm(pose[:3, 3:]) != 0:
                scale = self.uncertainty_aware_scale_recovery(inliers_ref,
                                                              inliers_cur,
                                                              pose,
                                                              pred_cur['depth'],
                                                              pred_cur['depthsigma'])
                rel_pose[:3, 3:] = pose[:3, 3:] * scale

            if np.linalg.norm(pose[:3, 3:]) == 0 or scale == -1:
                print('PnP '+str(i))
                pnp_pose = self.solve_pose_pnp(inliers_ref, inliers_cur, pred_ref['depth'])
                rel_pose = pnp_pose

            global_pose[:3, 3:] = np.matmul(global_pose[:3, :3], rel_pose[:3, 3:]) + global_pose[:3, 3:]
            global_pose[:3, :3] = np.matmul(global_pose[:3, :3], rel_pose[:3, :3])
            poses.append(copy.deepcopy(global_pose))
            pred_ref = copy.deepcopy(pred_cur)
        return poses
    
    def get_prediction(self, img, model):
        
        # img: [3, H, W] K: [3, 3]
        img = torch.from_numpy(np.transpose(img / 255.0, [2,0,1])).cuda().float().unsqueeze(0)
        
        infer = model.inference(img, self.top_points_ratio)

        pred = {'coord': infer['coord'].detach().squeeze(0).cpu().numpy(),
                'desc': infer['desc'].detach().squeeze(0).cpu().numpy(),
                'disp': infer['disp'].detach().squeeze().cpu().numpy(),
                'depth': infer['depth'].detach().squeeze().cpu().numpy(),
                'photosigma': infer['photosigma'].detach().squeeze().cpu(),
                'depthsigma': infer['depthsigma'].detach().squeeze().cpu()}
        return pred
    
    def bf_match(self, coord_ref, desc_ref, coord_cur, desc_cur):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc_cur, desc_ref)
        matches_idx = np.array([m.queryIdx for m in matches])
        matched_cur = coord_cur[matches_idx, :]
        matches_idx = np.array([m.trainIdx for m in matches])
        matched_ref = coord_ref[matches_idx, :]

        return matched_ref, matched_cur

    def normalize_coord(self, xy, K):
        xy_norm = copy.deepcopy(xy)
        xy_norm[:,0] = (xy[:,0] - K[0,2]) / K[0,0]
        xy_norm[:,1] = (xy[:,1] - K[1,2]) / K[1,1]

        return xy_norm
    
    def uncertainty_aware_scale_recovery(self, xy1, xy2, pose, depth2, sigma2):
        # Align the translation scale according to triangulation depth
        # xy1, xy2: [N, 2] pose: [4, 4] depth2: [H, W]
        
        # Triangulation
        img_h, img_w = np.shape(depth2)[0], np.shape(depth2)[1]
        pose_inv = np.linalg.inv(pose)

        xy1_norm = self.normalize_coord(xy1, self.cam_intrinsics)
        xy2_norm = self.normalize_coord(xy2, self.cam_intrinsics)

        _, points2_tri = cv_triangulation(np.concatenate([xy1_norm, xy2_norm], axis=1), pose_inv)
        
        depth2_tri = projection(xy2, points2_tri, img_h, img_w)
        depth2_tri[depth2_tri < 0] = 0
        
        # Remove negative depths
        valid_mask = (depth2 > 0) * (depth2_tri > 0)
        depth_pred_valid = depth2[valid_mask].reshape(-1, 1)
        sigma_pred_valid = sigma2[valid_mask].reshape(-1, 1)
        depth_tri_valid = depth2_tri[valid_mask].reshape(-1, 1)
        
        if np.sum(valid_mask) > self.inlier_min_num:
            sigma_pred_valid_min = torch.min(sigma_pred_valid, dim=0, keepdim=True)[0]
            sigma_pred_valid_max = torch.max(sigma_pred_valid, dim=0, keepdim=True)[0]
            scaled_sigma_valid = (sigma_pred_valid - sigma_pred_valid_min) / ((sigma_pred_valid_max - sigma_pred_valid_min) + 1e-12)
            prob_valid = 1. - scaled_sigma_valid
            prob_valid = prob_valid.div(torch.norm(prob_valid, p=2, dim=0, keepdim=True)).numpy()
    
            scale = depth_pred_valid / (depth_tri_valid + 1e-12)
            scale = np.sum((scale * (prob_valid**2)), 0)
        else:
            print(np.sum(valid_mask))
            scale = -1

        return scale
    
    def solve_pose_pnp(self, xy1, xy2, depth1):
        # Use pnp to solve relative poses.
        # xy1, xy2: [N, 2] depth1: [H, W]

        img_h, img_w = np.shape(depth1)[0], np.shape(depth1)[1]
        
        # Ensure all the correspondences are inside the image.
        x_idx = (xy2[:, 0] >= 0) * (xy2[:, 0] < img_w)
        y_idx = (xy2[:, 1] >= 0) * (xy2[:, 1] < img_h)
        idx = y_idx * x_idx
        xy1 = xy1[idx]
        xy2 = xy2[idx]

        xy1_int = xy1.astype(np.int)
        sample_depth = depth1[xy1_int[:,1], xy1_int[:,0]]
        valid_depth_mask = (sample_depth < self.max_depth) * (sample_depth > self.min_depth)

        xy1 = xy1[valid_depth_mask]
        xy2 = xy2[valid_depth_mask]

        # Unproject to 3d space
        points1 = unprojection(xy1, sample_depth[valid_depth_mask], self.cam_intrinsics)

        # ransac
        best_rt = []
        max_inlier_num = 0
        max_ransac_iter = self.PnP_ransac_times
        
        for i in range(max_ransac_iter):
            if xy2.shape[0] > 4:
                flag, r, t, inlier = cv2.solvePnPRansac(objectPoints=points1,
                                                        imagePoints=xy2,
                                                        cameraMatrix=self.cam_intrinsics,
                                                        distCoeffs=None,
                                                        iterationsCount=self.PnP_ransac_iter,
                                                        reprojectionError=self.PnP_ransac_thre)
                if flag and inlier.shape[0] > max_inlier_num:
                    best_rt = [r, t]
                    max_inlier_num = inlier.shape[0]
        pose = np.eye(4)
        if len(best_rt) != 0:
            r, t = best_rt
            pose[:3,:3] = cv2.Rodrigues(r)[0]
            pose[:3,3:] = t
        pose = np.linalg.inv(pose)
        return pose
    
    def solve_pose(self, xy1, xy2):
        # Solve essential matrix to find relative pose from flow.
        # ransac
        best_rt = []
        max_inlier_num = 0
        max_ransac_iter = self.pose_ransac_times
        best_inliers = np.ones((xy1.shape[0])) == 1
        pp = (self.cam_intrinsics[0,2], self.cam_intrinsics[1,2])

        # flow magnitude
        for i in range(max_ransac_iter):
            E, inliers = cv2.findEssentialMat(xy2,
                                              xy1,
                                              focal=self.cam_intrinsics[0,0],
                                              pp=pp,
                                              method=cv2.RANSAC,
                                              prob=0.99,
                                              threshold=self.pose_ransac_thre)
            cheirality_cnt, R, t, _ = cv2.recoverPose(E,
                                                      xy2,
                                                      xy1,
                                                      focal=self.cam_intrinsics[0,0],
                                                      pp=pp)
            if inliers.sum() > max_inlier_num and cheirality_cnt > self.inlier_min_num:
                best_rt = [R, t]
                max_inlier_num = inliers.sum()
                best_inliers = inliers

        if len(best_rt) == 0:
            R = np.eye(3)
            t = np.zeros((3,1))
            best_rt = [R, t]
        
        nonzero_id = np.nonzero(best_inliers)
        inliers_xy1 = xy1[nonzero_id[0], :]
        inliers_xy2 = xy2[nonzero_id[0], :]
        avg_parallax = np.mean(np.linalg.norm(inliers_xy1 - inliers_xy2, axis=1))

        if avg_parallax < self.inlier_min_parallax:
            R = np.eye(3)
            t = np.zeros((3,1))
            best_rt = [R, t]
        
        R, t = best_rt
        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3,3:] = t
        return pose, inliers_xy1, inliers_xy2

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description="VO")
    arg_parser.add_argument('--gpu', type=int, default=0,
                            help='gpu id.')
    arg_parser.add_argument('--mode', type=str, default='keypoint_depth',
                            choices=['keypoint_depth'],
                            help='')
    arg_parser.add_argument('--dataset', type=str, default='kitti',
                            choices=['kitti'],
                            help='')
    arg_parser.add_argument('--traj_save_txt', type=str, default=None,
                            help='directory for saving results')
    arg_parser.add_argument('--sequences_root', type=str, default=None,
                            help='directory for test sequences')
    arg_parser.add_argument('--sequence', type=str, default='09',
                            help='Test sequence id.')
    arg_parser.add_argument('--pretrained_model', type=str, default=None,
                            help='directory for loading pretrained models')

    
    # Setting
    arg_parser.add_argument('--new_hw', type=int, nargs='+', default=[320, 1024],
                            help='')
    arg_parser.add_argument('--top_points_ratio', type=float, default=0.3,
                            help='')
    arg_parser.add_argument('--num_scales', type=int, default=1,
                            help='')
    arg_parser.add_argument('--max_depth', type=float, default=80.0,
                            help='')
    arg_parser.add_argument('--min_depth', type=float, default=0.0,
                            help='')
    arg_parser.add_argument('--inlier_min_num', type=float, default=30,
                            help='')
    arg_parser.add_argument('--inlier_min_parallax', type=float, default=1.0,
                            help='')
    arg_parser.add_argument('--pose_ransac_times', type=int, default=1,
                            help='')
    arg_parser.add_argument('--pose_ransac_thre', type=float, default=0.2,
                            help='')
    arg_parser.add_argument('--PnP_ransac_iter', type=int, default=1000,
                            help='')
    arg_parser.add_argument('--PnP_ransac_thre', type=float, default=1.0,
                            help='')
    arg_parser.add_argument('--PnP_ransac_times', type=int, default=5,
                            help='')

    config = arg_parser.parse_args()

    torch.cuda.set_device(config.gpu)
    model = KeypointDepthVO(config)
    model.cuda()

    print('Model Loading...')
    weights = torch.load(config.pretrained_model, map_location='cuda:{}'.format(config.gpu))
    model.load_state_dict(weights['model_state'], strict=False)
    model.eval()

    print('Model Loaded.')

    vo = KPDepthVO(config)
    print('Data Loading...')
    if config.dataset == "kitti":
        images = vo.load_kitti()
    print('Data Loaded. Total ' + str(len(images)) + ' images found.')

    if config.mode == 'keypoint_depth':
        poses = vo.process_video(images, model)
    print('Test completed.')

    traj_txt = config.traj_save_txt
    save_traj(traj_txt, poses)
