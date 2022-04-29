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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import imgviz
import torch
from torch.utils.data import Dataset
import random

import yaml
from dataset import DsetBrain
from dataset import NUM_CLASSES, PIXEL_LIMIT, IMG_SIZE
from typing import Union
IMG_SIZE = tuple(IMG_SIZE)
MODEL_PATH = 'Train-BASE1-dset3-TEST'

dset_path = {
    '3_train': '/home/yhkim/workspace/project-CH-Labeling/labled_dset3_220426/trainset/**/img_mask/*.??g',
    '3_valid': '/home/yhkim/workspace/project-CH-Labeling/labled_dset3_220426/validset/**/img_mask/*.??g'
}

# softmax
def plot_dataset_prediction(model, data: Union[str, Dataset], idx: int, dtype: str='', show=False, save_path: str=None, single=True, threshold=0.5):
    retrain_flag = False
    is_dataset = False
    if isinstance(data, Dataset):
        is_dataset = True
        if data.is_train:
            data.is_train=False
            retrain_flag = True
    if is_dataset:
        img, truth_mask, org_img = data[idx]
    else:
        org_img = pad_to_square(cv2.imread(data, cv2.IMREAD_COLOR))
        if org_img.shape[:2]!=IMG_SIZE: org_img = cv2.resize(org_img, dsize=IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        img = torch.FloatTensor(org_img/255.0).permute(2, 0, 1)
        truth_mask = np.zeros(IMG_SIZE)
    
    model.eval()
    pred_softmax = multi_prediction(model, img, org_img, single=single)
    # pred = model(img.unsqueeze(0).cuda()).detach().squeeze()
    # pred_softmax = torch.nn.functional.softmax(pred, dim=0).cpu()
    img = img.permute(1, 2, 0).numpy()
    values, pred_mask = torch.max(pred_softmax, dim=0)
    pred_mask[values<threshold] = 0
    pred_mask = pred_mask.numpy().astype(np.uint8)
    vmax = NUM_CLASSES-1

    cols = 3
    rows = (NUM_CLASSES+3)//cols if (NUM_CLASSES+3)%cols==0 else (NUM_CLASSES+3)//cols+1
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*5, rows*5))
    
    idx = idx if is_dataset else ''
    
    plot_idx = 0
    # org image
    ax.ravel()[plot_idx].imshow(img)
    ax.ravel()[plot_idx].set_axis_off()
    ax.ravel()[plot_idx].set_title(f'{dtype}-{idx}-img')
    
    plot_idx += 1
    # truth mask
    if is_dataset:
        labeld_truth_mask = imgviz.label2rgb(truth_mask, label_names=LABEL_NAMES, loc="rb", font_size=FONT_SIZE)
        tm_plot = ax.ravel()[plot_idx].imshow(labeld_truth_mask, vmin=0, vmax=vmax)
        ax.ravel()[plot_idx].set_axis_off()
        ax.ravel()[plot_idx].set_title(f'{dtype}-{idx}-truth')
    else: ax.ravel()[plot_idx].set_axis_off()
    
    plot_idx += 1
    # threshold prediction
    pred_mask = (imgviz.label2rgb(pred_mask, label_names=LABEL_NAMES, loc="rb", font_size=FONT_SIZE)/255.0).astype(np.float32)
    thres_plot = ax.ravel()[plot_idx].imshow(pred_mask, vmin=0, vmax=vmax)
    ax.ravel()[plot_idx].set_axis_off()
    ax.ravel()[plot_idx].set_title(f'{dtype}-{idx}-threshold-{threshold}-prediction')
    
    plot_idx += 1
    # weighted image
    ax.ravel()[plot_idx].imshow(np.clip(cv2.addWeighted(img, 0.5, pred_mask, 1, 0), 0, 1))
    ax.ravel()[plot_idx].set_axis_off()
    ax.ravel()[plot_idx].set_title(f'{dtype}-{idx}-threshold-{threshold}-overlay')
    
    plot_idx += 1
    # class softmax prediction
    for i in range(plot_idx, plot_idx+NUM_CLASSES-1):
        softmax_plot = ax.ravel()[i].imshow(pred_softmax[i-plot_idx+1], vmin=0, vmax=1)
        plt.colorbar(softmax_plot, ax=ax.ravel()[i], shrink=SHRINK)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(f'{dtype}-{idx}-cls{i-plot_idx+1}-prediction')
    
    while i<cols*rows-1:
        i += 1
        ax.ravel()[i].set_axis_off()
    
    plt.tight_layout()
    
    if save_path: plt.savefig(save_path, dpi=rows*cols*20)
    if show: plt.show()
    else: plt.close()
    
    if retrain_flag:
        data.is_train=True

def validate_model(model, dataloader):
    loss_list = []
    with torch.no_grad():
        model.eval()
        for i, (image_batch, label_batch, _) in enumerate(dataloader, 1):
            print(f'\rValidation: {i}/{len(dataloader)}', end=' ')
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss_list.append(loss.item())
            
    return np.mean(loss_list).item()

def plot_losses(train_loss_list, valid_loss_list, show=False, save_path=None):
    plt.figure(figsize=(15, 6))
    plt.title('Train & Valid loss plot')
    plt.xlabel('iter_num')
    plt.ylabel('losses(log10)')
    plt.plot(list(train_loss_list.keys()), np.log10(list(train_loss_list.values())), label='Train_loss')
    plt.plot(list(valid_loss_list.keys()), np.log10(list(valid_loss_list.values())), label='Valid_loss')
    plt.legend()
    if save_path: plt.savefig(save_path, dpi=80)
    
    if show: plt.show()
    else: plt.close()

import numpy as np
import torch
import torch.optim as optim
import os
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from TransUNet.utils import DiceLoss
from torch.nn.modules.loss import CrossEntropyLoss
import logging
import sys
import gc
from torch import optim
from os.path import join
from datetime import datetime
from torch.nn.modules.loss import CrossEntropyLoss
from TransUNet.utils import DiceLoss

if __name__ == '__main__':
    if os.path.exists(MODEL_PATH):
        raise Exception('Folder already exists!')
    else:
        os.mkdir(MODEL_PATH)

    random.seed(777)
    train_mask_list = sorted(glob(dset_path['3_train'], recursive=True))
    val_mask_list = sorted(glob(dset_path['3_valid'], recursive=True))

    trainset = DsetBrain(train_mask_list, is_train=True)
    validset = DsetBrain(val_mask_list)

    from typing import Union
    from utils import multi_prediction, pad_to_square

    FONT_SIZE = 20
    SHRINK = 0.6
    LABEL_NAMES = ['0:bg', '1:ICH', '2:IVH', '3:EXTRA', '4:SAH', '5:ISDH']
    vit_name = 'R50-ViT-B_16'
    img_size = 512
    vit_patches_size = 16
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = NUM_CLASSES
    config_vit.n_skip = 3

    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
    model.load_from(weights=np.load('TransUNet/vit_models/imagenet21k+imagenet2012_R50+ViT-B_16.npz'))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters : {pytorch_total_params:,}')

    gc.collect()
    torch.cuda.empty_cache()
        
    logging.basicConfig(filename=f"{MODEL_PATH}/train_log_{str(datetime.now()).replace(' ', '_')}.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(MODEL_PATH)

    base_lr = 5e-3
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(NUM_CLASSES)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=base_lr, weight_decay=0.0001, momentum=0.9)
    #optimizer = optim.AdamW(model.parameters(), lr=base_lr)

    ########### imgaug 사용시 num_workers=0 설정 ###########
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=6, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=12, shuffle=False, num_workers=0, drop_last=False)

    TARGET_ITER_NUM = 40000
    CHKPOINT_MIN_ITER = 5000
    TRAIN_IDX = 55
    VALID_IDX = 5
    PER_ITER = 100

    loss_list = []
    train_loss_list = {}
    valid_loss_list = {}
    iter_num = 0
    lr_ = base_lr

    import time
    start_time = time.time()
    time_per_epoch_list = []

    while iter_num<=TARGET_ITER_NUM:
        trainset.roll()
        for i, (image_batch, label_batch, _) in enumerate(train_loader, 1):
            model.train()
            # model input
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            
            # get loss
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss_list.append(loss.item())
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # lr update
            lr_ = base_lr * (1.0 - iter_num / TARGET_ITER_NUM) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
                
            print(f'\rIter: {iter_num}/{TARGET_ITER_NUM}, Batch: {i:03}/{len(train_loader)}, Loss: {np.mean(loss_list):.8f}, Lr: {lr_:.10f}', end='')
            iter_num += 1
            if iter_num>TARGET_ITER_NUM:
                break

            # checkpoint
            if iter_num%PER_ITER==0:
                with torch.no_grad():
                    # get losses
                    print()
                    train_loss = np.mean(loss_list).item()
                    loss_list.clear()
                    valid_loss = validate_model(model, valid_loader)

                    # save checkpoint
                    if iter_num>=CHKPOINT_MIN_ITER and train_loss<min(train_loss_list.values()):
                        torch.save(model.state_dict(), f'{MODEL_PATH}/chk_min_train_loss.pth')
                        logging.info(f'TRAIN MODEL SAVED at {iter_num} iteration.')
                    if iter_num>=CHKPOINT_MIN_ITER and valid_loss<min(valid_loss_list.values()):
                        torch.save(model.state_dict(), f'{MODEL_PATH}/chk_min_valid_loss.pth')
                        logging.info(f'VALID MODEL SAVED at {iter_num} iteration.')

                    # append losses
                    train_loss_list[iter_num] = train_loss
                    valid_loss_list[iter_num] = valid_loss

                    # sample test
                    train_path = join(MODEL_PATH, f'plot_train_{TRAIN_IDX}_{iter_num:04}.jpg')
                    valid_path = join(MODEL_PATH, f'plot_valid_{VALID_IDX}_{iter_num:04}.jpg')
                    plot_dataset_prediction(model, trainset, TRAIN_IDX, 'Trainset', show=False, save_path=train_path)
                    plot_dataset_prediction(model, validset, VALID_IDX, 'Validset', show=False, save_path=valid_path)

                    # plot losses
                    if len(train_loss_list)>1:
                        plot_path = join(MODEL_PATH, f'plot_losses_{iter_num:04}.jpg')
                        plot_losses(train_loss_list, valid_loss_list, show=False, save_path=plot_path)

                    # calculate iou
                    train_score = trainset.calc_dataset_metric(model, metric='dice')
                    str_train_score = str({key: round(train_score[key], 2) for key in train_score})
                    #str_train_score = "{'mean': -1, 1: -1, 2: -1, 3: -1, 4: -1}"
                    valid_score = validset.calc_dataset_metric(model, metric='dice')
                    str_valid_score = str({key: round(valid_score[key], 2) for key in valid_score})

                    # calculate remaining time
                    cost_time_per_iter = (time.time() - start_time) / PER_ITER
                    time_per_epoch_list.append(cost_time_per_iter)
                    cost_time_per_iter = np.mean(time_per_epoch_list).item()
                    remaining_time = round((TARGET_ITER_NUM-iter_num) * cost_time_per_iter)
                    h, m = divmod(remaining_time, 3600); m, s = divmod(m, 60)
                    remaining_time = f'{h:02}H{m:02}M'
                    start_time = time.time()

                    # save log
                    logging.info('Iter: {}/{}, Tloss: {:.6f}, Vloss: {:.6f}, Tdice: {}, Vdice: {}, Lr: {:.10f}, remain: {}'.format(
                    iter_num, TARGET_ITER_NUM, train_loss, valid_loss, str_train_score, str_valid_score , lr_, remaining_time))