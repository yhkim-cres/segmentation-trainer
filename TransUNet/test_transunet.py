import numpy as np
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


vit_name = 'R50-ViT-B_16'
img_size = 224
vit_patches_size = 16
config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = 3
config_vit.n_skip = 3

if vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
print(net)