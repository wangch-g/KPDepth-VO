## Introduction
This is the implementation of our ICRA 2024 paper "Self-Supervised Learning of Monocular Visual Odometry and Depth with Uncertainty-Aware Scale Consistency" and the extended version of TCSVT 2025 paper "KPDepth-VO: Self-Supervised Learning of Scale-Consistent Visual Odometry and Depth with Keypoint Features from Monocular Video".

## Demo

<div align="center">
<img src="./demo/demo.gif" width="600px" height="180px"  align=center />

Input (upper left); Matching (upper right); Depth (lower left); Depth uncertainty (lower right)

</div>

## Dependency
We use python 3.8.13/cuda 11.4/torch 1.10.0/torchvision 0.11.0/opencv 3.4.8 for training and evaluation.

## Data Preparation
### COCO
We recommend following the <a href="https://github.com/wangch-g/lanet">LANet</a> to prepare the COCO dataset for training the Keypoint Network.
### KITTI odometry
Download <a href="http://www.cvlibs.net/datasets/kitti/eval_odometry.php">KITTI Odometry dataset</a>, and unzip the dataset into the same directory, like:
```
dataset
  | poses
  | sequences
```

### KITTI depth
For KITTI depth, download KITTI raw dataset from the <a href="http://www.cvlibs.net/download.php?file=raw_data_downloader.zip">script</a> provided on the official website. The data structure should be:
```
raw_data
  | 2011_09_26
  | 2011_09_28
  | 2011_09_29
  | 2011_09_30
  | 2011_10_03
```

## Training
The training code will be released soon.

## Evaluation
We provide the pre-trained models on <a href="https://drive.google.com/drive/folders/1G_pfkbyPXAJFmyf8OFLbFbwnTp8do-pW?usp=sharing">Google drive</a>/<a href="https://pan.baidu.com/s/1a3gkLwXVcjHqAu6Mbp1faA?pwd=2m1y">Baidu drive</a> for evaluating on KITTI odometry and KITTI depth respectively ('v1' for conference version and 'full' for journal version).
### KITTI odometry

We provide the pose files predicted by the proposed system and the ground truth in "./results/kitti_odom/ours/" and "./results/kitti_odom/gt/" respectively for evaluation, run:
```
python ./evaluation/eval_odom.py --gt_txt ./results/kitti_odom/gt/09.txt --pred_txt ./results/kitti_odom/ours/09.txt --seq 09
```

For evaluating the pre-trained model, run:
```
python kpdepth_vo.py --gpu [gpu id] --pretrained_model [/path/to/saved/checkpoints] --traj_save_dir [/where/to/save/your/predicted/poses/] --sequences_root [/path/to/odometry/dataset/sequences/] --sequence [sequences id]

python ./evaluation/eval_odom.py --gt_txt [/path/to/odometry/dataset/poses/seq_id.txt] --pred_txt [/path/to/your/predicted/poses/seq_id.txt] --seq [sequences id]
```
### KITTI depth
Run the following commands to generate the ground truth files for testing in eigen split.
```
cd ./splits/eigen
python export_gt_depth.py --data_path /path/to/your/kitti/raw_data/root 
```
In the main directory, run:
```
 python eval_depth.py --gpu [gpu id] --pretrained_model [/path/to/saved/checkpoints] --raw_data_dir [/path/to/your/kitti/raw_data/root]
```

## License
The code is released under the [MIT license](LICENSE).


## Citation
Please use the following citation when referencing our work:
```
@InProceedings{icra/Wang2024,
    author    = {Changhao Wang and
                 Guanwen Zhang and
                 Wei Zhou},
    title     = {Self-Supervised Learning of Monocular Visual Odometry and Depth
                 with Uncertainty-Aware Scale Consistency},
    booktitle = {{IEEE} International Conference on Robotics and Automation, {ICRA}
                  2024, Yokohama, Japan, May 13 - 17, 2024},
    pages     = {3984--3990},
    publisher = {{IEEE}},
    year      = {2024}
}

@article{tcsvt/Wang2025,
    author    = {Changhao Wang and
                 Guanwen Zhang and
                 Zhengyun Cheng and
                 Wei Zhou},
    title     = {KPDepth-VO: Self-Supervised Learning of Scale-Consistent Visual Odometry and Depth with Keypoint Features from Monocular Video},
    journal   = {IEEE Transactions on Circuits and Systems for Video Technology},
    volume    = {35},
    number    = {6},
    pages     = {5762-5775},
    publisher = {{IEEE}},
    year      = {2025}
}
```


## Related Projects
https://github.com/nianticlabs/monodepth2

https://github.com/Huangying-Zhan/DF-VO

https://github.com/B1ueber2y/TrianFlow
