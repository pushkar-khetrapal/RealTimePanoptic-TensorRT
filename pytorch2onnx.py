## importing libraries
import warnings
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision.models.detection.image_list import ImageList
import time

from realtime_panoptic.models.rt_pano_net import RTPanoNet
from realtime_panoptic.config import cfg
import realtime_panoptic.data.panoptic_transform as P
from realtime_panoptic.utils.visualization import visualize_segmentation_image, visualize_detection_image
import matplotlib.pyplot as plt

cityscapes_colormap = np.array([
 [128,  64, 128],
 [244,  35, 232],
 [ 70,  70,  70],
 [102, 102, 156],
 [190, 153, 153],
 [153, 153, 153],
 [250 ,170,  30],
 [220, 220,   0],
 [107, 142,  35],
 [152, 251, 152],
 [ 70, 130, 180],
 [220,  20,  60],
 [255,   0,   0],
 [  0,   0, 142],
 [  0,   0,  70],
 [  0,  60, 100],
 [  0,  80, 100],
 [  0,   0, 230],
 [119,  11,  32],
 [  0,   0,   0]])

cityscapes_instance_label_name = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
warnings.filterwarnings("ignore", category=UserWarning)
cfg.merge_from_file('RealTimePanoptic-TensorRT/configs/demo_config.yaml')

## developing model
model = RTPanoNet(
    backbone=cfg.model.backbone, 
    num_classes=cfg.model.panoptic.num_classes,
    things_num_classes=cfg.model.panoptic.num_thing_classes,
    pre_nms_thresh=cfg.model.panoptic.pre_nms_thresh,
    pre_nms_top_n=cfg.model.panoptic.pre_nms_top_n,
    nms_thresh=cfg.model.panoptic.nms_thresh,
    fpn_post_nms_top_n=cfg.model.panoptic.fpn_post_nms_top_n,
    instance_id_range=cfg.model.panoptic.instance_id_range)
device = 'cuda'
model.to(device)

model.load_state_dict(torch.load('cvpr_realtime_pano_cityscapes_standalone_no_prefix.pth'))

output_names = ["output_0"] + ["output_%d" % i for i in range(1,7)]
input_names = ["input1"]

x = torch.randn(1, 3, 1024, 2048, requires_grad=True).to('cuda')
torch.onnx.export(model, x, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)