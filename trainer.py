import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import yaml
import cv2
from glob import glob
from tqdm import tqdm
from random import sample
from pprint import pformat
from os.path import join
from dataset import DsetBrain
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss


import torch
import numpy as np

from load_models import load_models

with open('config.yaml', 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

class SegmentationTrainer:
    def __init__(self, model_name, optimizer_name, scheduler_name, config, **kwargs):
        # init variables
        self.config = config
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name

        # Load model
        self.model = load_models(config['model'][self.model_name], **config['general'])
        
        # Load Dataset
        self.train_mask_list = glob(join(config['dataset']['trainset_path'], '**/img_mask/*.??g'), recursive=True)
        self.valid_mask_list = glob(join(config['dataset']['validset_path'], '**/img_mask/*.??g'), recursive=True)
        self.trainset = DsetBrain(self.train_mask_list, is_train=True, **config['dataset'], **config['general'])
        self.validset = DsetBrain(self.valid_mask_list, is_train=False, **config['dataset'], **config['general'])

        # Load dataloader
        self.train_loader, self.valid_loader, self.test_loader = self.load_dataloader(self.trainset, self.validset, None)

        # Load loss layer
        self.ce_loss, self.dice_loss = self.load_loss_layer()
        # Load optimizer
        self.optimizer = self.load_optimizer()
        # Load Scheduler
        self.scheduler = self.load_scheduler(self.optimizer)
        
    def __str__(self):
        return pformat(self.config)

    def load_dataloader(self, trainset, validset, testset):
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.config['trainer']['train_batch_size'],
                        shuffle=True, num_workers=0, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=self.config['trainer']['valid_batch_size'],
                        shuffle=False, num_workers=0, drop_last=False)
        test_loader = None

        return train_loader, valid_loader, test_loader

    def load_loss_layer(self):
        return CrossEntropyLoss(), DiceLoss(len(self.config['general']['class_list'])+1)

    def load_optimizer(self):
        optim_name = self.optimizer_name
        if optim_name == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=float(self.config['trainer']['base_lr']),
                                **self.config['trainer']['optimizer'][optim_name])
        elif optim_name == 'AdamW':
            return torch.optim.AdamW(self.model.parameters(), lr=float(self.config['trainer']['base_lr']),
                                **self.config['trainer']['optimizer'][optim_name])
        else:
            raise Exception(f"{optim_name} does not exists!")

    def load_scheduler(self, optimizer):
        scheduler_name = self.scheduler_name
        if scheduler_name == 'CosineAnnealingWarmRestarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                              **self.config['trainer']['scheduler'][scheduler_name])

    def train_step(self, image_batch, label_batch):
        self.model.train()
        # model input
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        outputs = self.model(image_batch)
        
        # get loss
        loss_ce = self.ce_loss(outputs, label_batch[:].long())
        loss_dice = self.dice_loss(outputs, label_batch, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

trainer = SegmentationTrainer('TransUnet', 'SGD', 'CosineAnnealingWarmRestarts', config)