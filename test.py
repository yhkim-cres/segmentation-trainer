import os
import yaml
from load_models import load_models

with open('config.yaml', 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

model = load_models(**config['general'], **config['model']['TransUnet'])