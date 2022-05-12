import os
import yaml

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= config['general']['cuda_device']

    from trainer import SegmentationTrainer
    trainer = SegmentationTrainer(model_name='TransUnet', optimizer_name='SGD',
                                scheduler_name='CosineAnnealingWarmRestarts', config=config)
    trainer.train()