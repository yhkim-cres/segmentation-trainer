import random
import torch
import numpy as np
import imgaug.augmenters as iaa
import cv2
import imgviz
import matplotlib.pyplot as plt
from typing import Union
from torch.utils.data import Dataset

FONT_SIZE = 20
SHRINK = 0.6

# softmax
def plot_dataset_prediction(model, data: Union[str, Dataset], idx: int, img_size, class_list, label_names, dtype: str='',
                            show=False, save_path: str=None, single=True, threshold=0.5, **kwargs):
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
        if org_img.shape[:2]!=img_size: org_img = cv2.resize(org_img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        img = torch.FloatTensor(org_img/255.0).permute(2, 0, 1)
        truth_mask = np.zeros(img_size)
    
    model.eval()
    pred_softmax = multi_prediction(model, img, org_img, single=single)
    # pred = model(img.unsqueeze(0).cuda()).detach().squeeze()
    # pred_softmax = torch.nn.functional.softmax(pred, dim=0).cpu()
    img = img.permute(1, 2, 0).numpy()
    values, pred_mask = torch.max(pred_softmax, dim=0)
    pred_mask[values<threshold] = 0
    pred_mask = pred_mask.numpy().astype(np.uint8)
    num_classes = len(class_list+1)
    vmax = num_classes-1

    cols = 3
    rows = (num_classes+3)//cols if (num_classes+3)%cols==0 else (num_classes+3)//cols+1
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
        labeld_truth_mask = imgviz.label2rgb(truth_mask, label_names=label_names, loc="rb", font_size=FONT_SIZE)
        tm_plot = ax.ravel()[plot_idx].imshow(labeld_truth_mask, vmin=0, vmax=vmax)
        ax.ravel()[plot_idx].set_axis_off()
        ax.ravel()[plot_idx].set_title(f'{dtype}-{idx}-truth')
    else: ax.ravel()[plot_idx].set_axis_off()
    
    plot_idx += 1
    # threshold prediction
    pred_mask = (imgviz.label2rgb(pred_mask, label_names=label_names, loc="rb", font_size=FONT_SIZE)/255.0).astype(np.float32)
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
    for i in range(plot_idx, plot_idx+num_classes-1):
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

def pad_to_square(img):
    if img.shape[0]==img.shape[1]:
        return img
    
    length = max(img.shape)
    delta_w = length - img.shape[1]
    delta_h = length - img.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    pad_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 255])
    
    return pad_img

def multi_prediction(model, img, org_img, single=True):
    pred = model(img.unsqueeze(0).cuda()).detach().squeeze()
    pred_softmax = torch.nn.functional.softmax(pred, dim=0)
    if single:
        return pred_softmax.cpu()
    
    aug_list = [iaa.Sharpen(0.8), iaa.GammaContrast(2), iaa.Fliplr(1)]
    for aug in aug_list:
        aug_img = aug.augment_image(org_img)
        img_tensor = torch.FloatTensor(aug_img/255.0).permute(2, 0, 1)
        pred_aug = model(img_tensor.unsqueeze(0).cuda()).detach().squeeze()
        if isinstance(aug, iaa.Fliplr):
            pred_aug = torch.flip(pred_aug, dims=(-1,))
        pred_softmax += torch.nn.functional.softmax(pred_aug, dim=0)
    
    pred_softmax = (pred_softmax/(len(aug_list)+1))
    
    return pred_softmax.cpu()

def multiply_list(lst: list, mul: float, seed=777):
    integer, point = divmod(mul, 1)
    random.seed(seed)
    lst = lst*int(integer) + random.sample(lst, round(point*len(lst)))
    
    return lst
    
def calc_iou(truth_mask, pred, class_list, pixel_limit, smooth=1e-5):
    iou_list = dict()
    # IOU
    for cls_idx in class_list:
        iou_list.setdefault(cls_idx, -1)
        if torch.sum(truth_mask==cls_idx)>pixel_limit:
            cls_and = (torch.sum(torch.logical_and(truth_mask==cls_idx, pred[cls_idx]))).item()
            cls_or = (torch.sum(torch.logical_or(truth_mask==cls_idx, pred[cls_idx]))).item()
            iou = (cls_and+smooth) / (cls_or+smooth)
            iou_list[cls_idx] = iou
    iou_values = [x for x in iou_list.values() if x!=-1]
    
    return iou_list, np.mean(iou_values).item() if iou_values else -1

def calc_dice(truth_mask, pred, class_list, pixel_limit, smooth=1e-5):
    dice_list = dict()
    # DICE
    for cls_idx in class_list:
        dice_list.setdefault(cls_idx, -1)
        if torch.sum(truth_mask==cls_idx)>pixel_limit:
            cls_truth_mask = truth_mask==cls_idx
            cls_and = (torch.sum(torch.logical_and(cls_truth_mask, pred[cls_idx]))).item()
            cls_x_y = torch.sum(cls_truth_mask).item() + torch.sum(pred[cls_idx]).item()
            dice = (cls_and*2+smooth) / (cls_x_y+smooth)
            dice_list[cls_idx] = dice
    dice_values = [x for x in dice_list.values() if x!=-1]
    
    return dice_list, np.mean(dice_values).item() if dice_values else -1