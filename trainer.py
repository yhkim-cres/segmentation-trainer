import yaml
import os
import time
import torch
import numpy as np
import logging
import sys
from load_models import load_models
from glob import glob
from pprint import pformat
from os.path import join
from dataset import DsetBrain
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss, plot_losses, plot_dataset_prediction
from datetime import datetime
class SegmentationTrainer:
    def __init__(self, model_name, optimizer_name, scheduler_name, config, test_mode=False, **kwargs):
        # Init variables
        self.config = config
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.test_mode = test_mode

        if not self.test_mode:
            # Start logging
            self.init_logging(self.config['trainer']['log_path'])

        # Load model
        self.model = load_models(config['model'][self.model_name], **config['general']).cuda()
        
        # Load Dataset
        self.train_mask_list = sorted(glob(join(config['dataset']['trainset_path'], '**/img_mask/*.??g'), recursive=True))
        self.valid_mask_list = sorted(glob(join(config['dataset']['validset_path'], '**/img_mask/*.??g'), recursive=True))
        is_train = True if not self.test_mode else False
        self.trainset = DsetBrain(self.train_mask_list, is_train=is_train, **config['dataset'], **config['general'])
        self.validset = DsetBrain(self.valid_mask_list, is_train=False, **config['dataset'], **config['general'])

        # Load dataloader
        self.train_loader, self.valid_loader, self.test_loader = self.load_dataloader(self.trainset, self.validset, None)

        # Load loss layer
        self.ce_loss, self.dice_loss = self.load_loss_layer()
        # Load optimizer
        self.optimizer = self.load_optimizer()
        # Load Scheduler
        self.scheduler = self.load_scheduler(self.optimizer)

        # init training variables
        self.iter_num = 0
        self.train_loss_list = {}
        self.valid_loss_list = {}
        
    def __str__(self):
        string = f"model: {self.model_name}, optimizer: {self.optimizer_name}, scheduler: {self.scheduler_name}"
        return string+'\n'+pformat(self.config)

    def init_logging(self, log_path):
        if os.path.exists(log_path):
            raise Exception('Folder already exists!')
        else:
            os.makedirs(log_path, exist_ok=False)
            
        logging.basicConfig(filename=f"{log_path}/train_log_{str(datetime.now()).replace(' ', '_')}.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(log_path)
        logging.info(str(self))

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
            return torch.optim.SGD(self.model.parameters(), lr=self.config['trainer']['base_lr'],
                                **self.config['trainer']['optimizer'][optim_name])
        elif optim_name == 'AdamW':
            return torch.optim.AdamW(self.model.parameters(), lr=self.config['trainer']['base_lr'],
                                **self.config['trainer']['optimizer'][optim_name])
        else:
            raise Exception(f"{optim_name} does not exists!")

    def load_scheduler(self, optimizer):
        scheduler_name = self.scheduler_name
        if scheduler_name == 'CosineAnnealingWarmRestarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                              **self.config['trainer']['scheduler'][scheduler_name])

    def train_step(self, image_batch, label_batch):
        ce_value, dice_value = self.config['trainer']['loss_value']['ce_loss'], self.config['trainer']['loss_value']['dice_loss']
        self.model.train()
        # model input
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        outputs = self.model(image_batch)
        
        # get loss
        loss_ce = self.ce_loss(outputs, label_batch[:].long())
        loss_dice = self.dice_loss(outputs, label_batch, softmax=True)
        loss = ce_value*loss_ce + dice_value*loss_dice

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_step(self):
        loss_list = []
        with torch.no_grad():
            self.model.eval()
            for i, (image_batch, label_batch, _) in enumerate(self.valid_loader, 1):
                print(f'\rValidation: {i}/{len(self.valid_loader)}', end=' ')
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = self.model(image_batch)
                loss_ce = self.ce_loss(outputs, label_batch[:].long())
                loss_dice = self.dice_loss(outputs, label_batch, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                loss_list.append(loss.item())
                
        return np.mean(loss_list).item()

    def dataset_sample_prediction(self, log_path, train_sample_idx, valid_sample_idx):
        train_path = join(log_path, f'plot_train_{train_sample_idx}_{self.iter_num:04}.jpg')
        valid_path = join(log_path, f'plot_valid_{valid_sample_idx}_{self.iter_num:04}.jpg')
        plot_dataset_prediction(self.model, self.trainset, train_sample_idx, dtype='Trainset',
                            show=False, save_path=train_path, **self.config['general'])
        plot_dataset_prediction(self.model, self.validset, valid_sample_idx, dtype='Validset',
                            show=False, save_path=valid_path, **self.config['general'])

    def dataset_calc_metric(self, threshold, metric='dice'):
        train_score = self.trainset.calc_dataset_metric(self.model, metric=metric, threshold=threshold)
        str_train_score = str({key: round(train_score[key], 2) for key in train_score})
        #str_train_score = "{'mean': -1, 1: -1, 2: -1, 3: -1, 4: -1}"
        valid_score = self.validset.calc_dataset_metric(self.model, metric=metric, threshold=threshold)
        str_valid_score = str({key: round(valid_score[key], 2) for key in valid_score})

        return str_train_score, str_valid_score

    def train(self):
        # config variables
        trainer_config = self.config['trainer']
        log_path = trainer_config['log_path']
        target_iteration = trainer_config['target_iteration']
        min_chkpoint_iteration = trainer_config['min_chkpoint_iteration']
        per_log_iter = trainer_config['per_log_iter']
        train_sample_idx = trainer_config['train_sample_idx']
        valid_sample_idx = trainer_config['valid_sample_idx']
        threshold = self.config['general']['threshold']

        # set train variables
        start_time = time.time()
        time_per_epoch_list = []
        loss_list = []

        while self.iter_num <= target_iteration:
            self.trainset.roll()
            for i, (image_batch, label_batch, _) in enumerate(self.train_loader, 1):
                # train step
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                loss = self.train_step(image_batch, label_batch)
                loss_list.append(loss)

                # lr update
                lr_ = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()

                print(f'\rIter: {self.iter_num}/{target_iteration}, Batch: {i:03}/{len(self.train_loader)}, Loss: {np.mean(loss_list):.8f}, Lr: {lr_:.10f}', end='')
                self.iter_num += 1
                if self.iter_num>target_iteration:
                    break

                # validation step
                if self.iter_num%per_log_iter==0:
                    with torch.no_grad():
                        # get losses
                        print()
                        train_loss = np.mean(loss_list).item()
                        loss_list.clear()
                        valid_loss = self.validation_step()

                        # save checkpoint
                        if self.iter_num>=min_chkpoint_iteration and valid_loss<min(self.valid_loss_list.values()):
                            torch.save(self.model.state_dict(), f'{log_path}/chk_min_valid_loss.pth')
                            logging.info(f'VALID MODEL SAVED at {self.iter_num} iteration.')

                        # append losses
                        self.train_loss_list[self.iter_num] = train_loss
                        self.valid_loss_list[self.iter_num] = valid_loss

                        # sample test
                        self.dataset_sample_prediction(log_path, train_sample_idx, valid_sample_idx)

                        # plot losses
                        if len(self.train_loss_list)>1:
                            plot_path = join(log_path, f'plot_losses_{self.iter_num:04}.jpg')
                            plot_losses(self.train_loss_list, self.valid_loss_list, show=False, save_path=plot_path)

                        # calculate iou
                        str_train_score, str_valid_score = self.dataset_calc_metric(threshold=threshold, metric='dice')

                        # calculate remaining time
                        cost_time_per_iter = (time.time() - start_time) / per_log_iter
                        time_per_epoch_list.append(cost_time_per_iter); cost_time_per_iter = np.mean(time_per_epoch_list).item()
                        remaining_time = round((target_iteration-self.iter_num) * cost_time_per_iter)
                        h, m = divmod(remaining_time, 3600); m, s = divmod(m, 60)
                        remaining_time = f'{h:02}H{m:02}M'; start_time = time.time()

                        # save log
                        logging.info('Iter: {}/{}, Tloss: {:.6f}, Vloss: {:.6f}, Tdice: {}, Vdice: {}, Lr: {:.10f}, remain: {}'.format(
                        self.iter_num, target_iteration, train_loss, valid_loss, str_train_score, str_valid_score , lr_, remaining_time))

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= config['general']['cuda_device']
    trainer = SegmentationTrainer('TransUnet', 'SGD', 'CosineAnnealingWarmRestarts', config)
    trainer.train()