import random
import yaml
import torch
import numpy as np
import imgaug.augmenters as iaa
import cv2

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
    
def calc_iou(truth_mask, pred, num_classes, pixel_limit, smooth=1e-5):
    iou_list = dict()
    # IOU
    for cls_idx in range(1, num_classes):
        iou_list.setdefault(cls_idx, -1)
        if torch.sum(truth_mask==cls_idx)>pixel_limit:
            cls_and = (torch.sum(torch.logical_and(truth_mask==cls_idx, pred[cls_idx]))).item()
            cls_or = (torch.sum(torch.logical_or(truth_mask==cls_idx, pred[cls_idx]))).item()
            iou = (cls_and+smooth) / (cls_or+smooth)
            iou_list[cls_idx] = iou
    iou_values = [x for x in iou_list.values() if x!=-1]
    
    return iou_list, np.mean(iou_values).item() if iou_values else -1

def calc_dice(truth_mask, pred, num_classes, pixel_limit, smooth=1e-5):
    dice_list = dict()
    # DICE
    for cls_idx in range(1, num_classes):
        dice_list.setdefault(cls_idx, -1)
        if torch.sum(truth_mask==cls_idx)>pixel_limit:
            cls_truth_mask = truth_mask==cls_idx
            cls_and = (torch.sum(torch.logical_and(cls_truth_mask, pred[cls_idx]))).item()
            cls_x_y = torch.sum(cls_truth_mask).item() + torch.sum(pred[cls_idx]).item()
            dice = (cls_and*2+smooth) / (cls_x_y+smooth)
            dice_list[cls_idx] = dice
    dice_values = [x for x in dice_list.values() if x!=-1]
    
    return dice_list, np.mean(dice_values).item() if dice_values else -1