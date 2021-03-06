import cv2
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset
from image_augmentation import ImgAug
from utils import multi_prediction, calc_dice, calc_iou, oversample
class DsetBrain(Dataset):
    def __init__(self, mask_list, class_list, img_shape, pixel_limit, oversampling_values, non_label_ratio, 
                is_train=False, **kwargs):
        """
        0 : Background
        1 : ICH
        2 : IVH
        3 : extra
        4 : sah
        """
        self.mask_list = mask_list  # mask image path
        self.img_list = [x.replace('/img_mask/', '/img/') for x in mask_list]  # image path
        self.img_aug = ImgAug()  # image augmentation instance
        self.is_train = is_train  # if is_train : augmentation & oversampling
        self.class_idx_list = {}  # idx_list for each classes

        self.class_list = class_list
        self.img_shape = tuple(img_shape)
        self.pixel_limit = pixel_limit
        self.non_label_ratio = non_label_ratio
        self.oversampling_values = oversampling_values
        
        self.class_idx_list.setdefault(0, [])
        for cls_idx in self.class_list:
            self.class_idx_list.setdefault(cls_idx, [])

        # sort classes according to PIXEL LIMIT
        for i in tqdm(range(len(self.mask_list)), desc='Sorting claases'):
            mask = cv2.imread(self.mask_list[i], 0)
            is_bg = True
            for cls_idx in self.class_list:
                pixel_num = np.sum(mask==cls_idx).item()
                if pixel_num>self.pixel_limit:
                    self.class_idx_list[cls_idx].append(i)
                    is_bg = False
            if is_bg:
                self.class_idx_list[0].append(i)

        self.org_idx_list = []  # original idx list
        self.train_idx_list = []  # oversampled idx list
        self.roll()
        
    def roll(self):  # roll training list
        self.train_idx_list = []  # oversampled idx list
        # ch list oversampling
        not_bg_list = []
        for key, values in self.class_idx_list.items():
            if key==0: continue
            if self.is_train and key in self.oversampling_values:
                not_bg_list += oversample(values, self.oversampling_values[key])
            else:
                not_bg_list += values
        self.train_idx_list += not_bg_list

        # non_ch sampling
        self.train_idx_list += random.sample(self.class_idx_list[0], min(len(not_bg_list), round(len(self.class_idx_list[0])*self.non_label_ratio)))

    def __len__(self):
        return len(self.train_idx_list) if self.is_train else len(self.org_idx_list)
    
    def __getitem__(self, idx):
        idx = self.train_idx_list[idx] if self.is_train else self.org_idx_list[idx]
        
        # Read Image
        img = cv2.imread(self.img_list[idx], 1)
        if img.shape[:2]!=self.img_shape:
            img = cv2.resize(img, dsize=self.img_shape, interpolation=cv2.INTER_NEAREST)
        
        # Read Mask
        mask = cv2.imread(self.mask_list[idx], 0)
        if mask.shape[:2]!=self.img_shape:
            mask = cv2.resize(mask, dsize=self.img_shape, interpolation=cv2.INTER_NEAREST)
        
        # Create label mask
        label_mask = np.zeros(mask.shape, dtype=np.uint8)
        for cls_idx in self.class_list:
            if idx in self.class_idx_list[cls_idx]:
                label_mask[mask==cls_idx] = cls_idx

        # Apply Augmentation
        if self.is_train:
            img, label_mask = self.img_aug.apply_aug(img, label_mask)

        return torch.FloatTensor(img/255.0).permute(2, 0, 1), torch.LongTensor(label_mask), torch.FloatTensor(img/255.0)
    
    def train(self):
        self.is_train = True
    
    def eval(self):
        self.is_train = False
        
    def plot_class_distribution(self, is_train=False):
        class_idx_list = self.class_idx_list
        len_list = list(map(len, class_idx_list.values()))
        print(len_list)
        plt.figure(figsize=(10, 7))
        plt.bar(class_idx_list.keys(), len_list)
        plt.title('Class Count Barplot', fontsize=20)
        plt.xlabel('class')
        plt.ylabel('count')
        plt.show()
    
    def plot_sample(self, idx, save_path=None, show=True):
        img, truth_mask, _ = self[idx]

        fig = plt.figure(figsize=(12, 6))
        rows, cols = 1, 2
        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.set_title(f'{idx}-img')
        ax1.imshow(img.permute(1, 2, 0))

        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.set_title(f'{idx}-truth')
        tm_plot = ax2.imshow(truth_mask, vmin=0, vmax=len(self.class_list))
        plt.colorbar(tm_plot, shrink=0.6)
        
        if save_path: fig.savefig(save_path, dpi=100)
        if show: plt.show()
        else: plt.close()
        
    def calc_dataset_metric(self, model, metric: str='dice', single=True, smooth=1e-5, threshold=0.5):
        retrain_flag = False
        if self.is_train:
            self.is_train=False
            retrain_flag = True
            
        if metric=='dice': metric_function = calc_dice
        elif metric=='iou': metric_function = calc_iou
        else: raise Exception(f'{metric} is not included in the metric functions.')
        
        model.eval()
        class_score_dict = dict()
        class_score_dict['mean'] = []
        for cls_idx in self.class_list:
            class_score_dict.setdefault(cls_idx, [])
        
        idx_list = set(chain(*[self.class_idx_list[key] for key in self.class_idx_list if key!=0]))
        for idx in tqdm(idx_list, desc=f'Calc {metric}'):
            img, truth_mask, org_img = self[idx]
            pred_softmax = multi_prediction(model, img, org_img, single=single)
            values, pred_mask = torch.max(pred_softmax, dim=0)
            pred_mask[values<threshold] = 0
            
            score_list, score_mean = metric_function(truth_mask, pred_mask, self.class_list, self.pixel_limit)
            for key in score_list:
                if score_list[key]!=-1: class_score_dict[key].append(score_list[key])
            if score_mean!=-1: class_score_dict['mean'].append(score_mean)
            
        if retrain_flag:
            self.is_train = True
        
        for key in class_score_dict:
            class_score_dict[key] = np.mean(class_score_dict[key]).item()
        
        return class_score_dict
    
    # ????????????
    def calc_dataset_performance(self, model, threshold=0.5):
        retrain_flag = False
        if self.is_train:
            self.is_train=False
            retrain_flag = True
            
        model.eval()
        pred_idx_list = dict()
        
        idx_list = set(chain(*[self.class_idx_list[key] for key in self.class_idx_list]))
        for idx in tqdm(sorted(idx_list), desc=f'Calc Performance'):
            pred_idx_list.setdefault(idx, [])
            img, truth_mask, _ = self[idx]
            pred = model(img.unsqueeze(0).cuda()).detach().squeeze().cpu()
            pred = torch.nn.functional.softmax(pred, dim=0) >= threshold
            
            truth_cls, pred_cls = [], []
            for cls_num in self.class_list:
                if torch.sum(truth_mask==cls_num) > self.pixel_limit:
                    truth_cls.append(cls_num)
                if torch.sum(pred[cls_num]) > self.pixel_limit:
                    pred_cls.append(cls_num)
            if not truth_cls:
                truth_cls.append(0)
            if not pred_cls:
                pred_cls.append(0)
            pred_idx_list[idx].append(truth_cls)
            pred_idx_list[idx].append(pred_cls)
        
        scores = {}
        for cls_num in [0]+self.class_list:
            scores.setdefault(cls_num, [])
            tp, fp, fn, tn = 0, 0, 0, 0
            for key in pred_idx_list:
                if cls_num in pred_idx_list[key][0] and cls_num in pred_idx_list[key][1]:
                    tp += 1
                elif cls_num not in pred_idx_list[key][0] and cls_num in pred_idx_list[key][1]:
                    fp += 1
                elif cls_num in pred_idx_list[key][0] and cls_num not in pred_idx_list[key][1]:
                    fn += 1
                elif cls_num not in pred_idx_list[key][0] and cls_num not in pred_idx_list[key][1]:
                    tn += 1
            acc = (tp+tn)/(tp+fp+fn+tn)
            sensitivity = tp/(tp+fn)
            specificity = tn/(tn+fp)
            scores[cls_num].extend([acc, sensitivity, specificity])
                    
        if retrain_flag:
            self.is_train = True
        
        return scores

import pydicom
from utils import window_image
class DsetDcm(Dataset):
    def __init__(self, mask_list, class_list, img_shape, pixel_limit, oversampling_values, non_label_ratio,
                window_center=50, window_width=100, is_train=False, **kwargs):
        """
        0 : Background
        1 : ICH
        2 : IVH
        3 : extra
        4 : sah
        """
        self.mask_list = mask_list  # mask image path
        self.dcm_list = [x.replace('/img_mask/', '/dcm/').replace('.png', '.dcm') for x in mask_list]  # image path
        self.img_aug = ImgAug(only_affine=False)  # image augmentation instance
        self.is_train = is_train  # if is_train : augmentation & oversampling
        self.class_idx_list = {}  # idx_list for each classes

        self.class_list = class_list
        self.img_shape = tuple(img_shape)
        self.pixel_limit = pixel_limit
        self.non_label_ratio = non_label_ratio
        self.oversampling_values = oversampling_values
        self.window_center = window_center
        self.window_width = window_width
        
        self.class_idx_list.setdefault(0, [])
        for cls_idx in self.class_list:
            self.class_idx_list.setdefault(cls_idx, [])

        # sort classes according to PIXEL LIMIT
        for i in tqdm(range(len(self.mask_list)), desc='Sorting claases'):
            mask = cv2.imread(self.mask_list[i], 0)
            is_bg = True
            for cls_idx in self.class_list:
                if cls_idx==0: continue
                pixel_num = np.sum(mask==cls_idx).item()
                if pixel_num>self.pixel_limit:
                    self.class_idx_list[cls_idx].append(i)
                    is_bg = False
            if is_bg:
                self.class_idx_list[0].append(i)

        self.org_idx_list = []  # original idx list
        self.train_idx_list = []  # oversampled idx list
        self.roll()

    @staticmethod
    def get_image_from_dcm(dcm_path, window_center=50, window_width=100):
        dcm_file = pydicom.read_file(dcm_path, force=True)
        dcm_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        image = dcm_file.pixel_array
        intercept = dcm_file.RescaleIntercept
        slope = dcm_file.RescaleSlope
        hu_image = image * slope + intercept

        return window_image(hu_image, window_center, window_width)
        
    def roll(self):  # roll training list
        self.train_idx_list = []  # oversampled idx list
        self.org_idx_list = []

        # label list oversampling
        label_list = []
        for key, values in self.class_idx_list.items():
            if key==0: continue
            if self.is_train and key in self.oversampling_values:
                label_list += oversample(values, self.oversampling_values[key])
            else:
                label_list += values
        
        non_label_train_list = random.sample(self.class_idx_list[0], min(len(self.class_idx_list[0]), round(len(label_list)*self.non_label_ratio)))
        non_label_org_idx_list = random.sample(self.class_idx_list[0], min(len(self.class_idx_list[0]), round(len(set(label_list))*self.non_label_ratio)))

        # non_label sampling
        self.org_idx_list = list(set(label_list + non_label_org_idx_list))
        self.train_idx_list = label_list + non_label_train_list

    def __len__(self):
        return len(self.train_idx_list) if self.is_train else len(self.org_idx_list)
    
    def __getitem__(self, idx):
        idx = self.train_idx_list[idx] if self.is_train else self.org_idx_list[idx]
        
        # Read dcm
        dcm_file = pydicom.read_file(self.dcm_list[idx], force=True)
        dcm_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        image = dcm_file.pixel_array
        intercept = dcm_file.RescaleIntercept
        slope = dcm_file.RescaleSlope
        hu_image = image * slope + intercept
        if hu_image.shape[:2]!=self.img_shape:
            hu_image = cv2.resize(hu_image, dsize=self.img_shape, interpolation=cv2.INTER_CUBIC)

        # Get window image
        hu_image = window_image(hu_image, self.window_center, self.window_width)
        rgb_img = cv2.cvtColor(hu_image, cv2.COLOR_GRAY2RGB)
        
        # Read Mask
        mask = cv2.imread(self.mask_list[idx], 0)
        if mask.shape[:2]!=self.img_shape:
            mask = cv2.resize(mask, dsize=self.img_shape, interpolation=cv2.INTER_NEAREST)
        
        # Create label mask
        label_mask = np.zeros(mask.shape, dtype=np.uint8)
        for cls_idx in self.class_list:
            if cls_idx==0: continue
            if idx in self.class_idx_list[cls_idx]:
                label_mask[mask==cls_idx] = cls_idx

        # Apply Augmentation
        if self.is_train:
            hu_image, label_mask = self.img_aug.apply_aug(hu_image, label_mask)

        return torch.FloatTensor(hu_image/255.0).unsqueeze(0), torch.LongTensor(label_mask), torch.FloatTensor(rgb_img/255.0)
    
    def train(self):
        self.is_train = True
    
    def eval(self):
        self.is_train = False
        
    def plot_class_distribution(self, is_train=False):
        class_idx_list = self.class_idx_list
        len_list = list(map(len, class_idx_list.values()))
        print(len_list)
        plt.figure(figsize=(10, 7))
        plt.bar(class_idx_list.keys(), len_list)
        plt.title('Class Count Barplot', fontsize=20)
        plt.xlabel('class')
        plt.ylabel('count')
        plt.show()
    
    def plot_sample(self, idx, save_path=None, show=True):
        hu_image, truth_mask, img = self[idx]

        fig = plt.figure(figsize=(12, 6))
        rows, cols = 1, 2
        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.set_title(f'{idx}-img')
        ax1.imshow(img)

        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.set_title(f'{idx}-truth')
        tm_plot = ax2.imshow(truth_mask, vmin=0, vmax=len(self.class_list))
        plt.colorbar(tm_plot, shrink=0.6)
        
        if save_path: fig.savefig(save_path, dpi=100)
        if show: plt.show()
        else: plt.close()
        
    def calc_dataset_metric(self, model, metric: str='dice', single=True, smooth=1e-5, threshold=0.5):
        retrain_flag = False
        if self.is_train:
            self.is_train=False
            retrain_flag = True
            
        if metric=='dice': metric_function = calc_dice
        elif metric=='iou': metric_function = calc_iou
        else: raise Exception(f'{metric} is not included in the metric functions.')
        
        model.eval()
        class_score_dict = dict()
        class_score_dict['mean'] = []
        for cls_idx in self.class_list:
            class_score_dict.setdefault(cls_idx, [])
        
        # idx_list = set(chain(*[self.class_idx_list[key] for key in self.class_idx_list if key!=0]))
        idx_num = len(self.org_idx_list)
        for idx in tqdm(range(idx_num), desc=f'Calc {metric}'):
            img, truth_mask, org_img = self[idx]
            pred_softmax = multi_prediction(model, img, org_img, single=single)
            values, pred_mask = torch.max(pred_softmax, dim=0)
            pred_mask[values<threshold] = 0
            
            score_list, score_mean = metric_function(truth_mask, pred_mask, self.class_list, self.pixel_limit)
            for key in score_list:
                if score_list[key]!=-1: class_score_dict[key].append(score_list[key])
            if score_mean!=-1: class_score_dict['mean'].append(score_mean)
            
        if retrain_flag:
            self.is_train = True
        
        for key in class_score_dict:
            class_score_dict[key] = np.mean(class_score_dict[key]).item()
        
        return class_score_dict