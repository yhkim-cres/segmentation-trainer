import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help='config yaml path', default='config.yaml')
parser.add_argument("-m", "--model", help='model name', default='TransUnet')
parser.add_argument("-o", "--optimizer", help='optimizer name', default='AdamW')
parser.add_argument("-s", "--scheduler", help='scheduler_name', default='CosineAnnealingWarmRestarts')
args = parser.parse_args()

if __name__ == '__main__':
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= config['general']['cuda_device']
    print(args)
    from trainer import SegmentationTrainer
    trainer = SegmentationTrainer(model_name=args.model, optimizer_name=args.optimizer,
                                scheduler_name=args.scheduler, test_mode=False, config=config)
    trainer.train()