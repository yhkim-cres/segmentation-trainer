import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from random import sample
from image_augmentation import ImgAug
from itertools import chain
import os
import yaml
from os.path import join
from dataset import DsetBrain
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

from load_models import load_models

with open('config.yaml', 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

#model = load_models(**config['general'], **config['model']['TransUnet'])

train_mask_list = glob(join(config['dataset']['trainset_path'], '**/img_mask/*.??g'), recursive=True)
valid_mask_list = glob(join(config['dataset']['validset_path'], '**/img_mask/*.??g'), recursive=True)

# trainset = DsetBrain(train_mask_list, is_train=True, **config['general'], **config['dataset'])
# validset = DsetBrain(valid_mask_list, is_train=False, **config['general'], **config['dataset'])