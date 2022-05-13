# segmentation-trainer
Crescom Segmentation Trainer Implementation for CH segmentation<br>

## Environment
- Ubuntu 20.04 LTS
- Miniconda 가상환경
- 3.7≤python≤3.8
- cuda≥11.1, cudnn 설치

## Installation
```bash
conda create -n transunet python=3.8
conda activate transunet

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install -c conda-forge tqdm matplotlib
pip install opencv-contrib-python ml-collections medpy imgaug imgviz timm einops

# jupyterlab 설치
conda install -c conda-forge jupyterlab ipywidgets

# jupyterlab 실행
jupyter lab
```

## Train
1. ```config.yaml```에서 hyperparameter 설정
2. ```python run_trainer.py --config config.yaml --model TransUnet --optimizer SGD --scheduler CosineAnnealingWarmRestarts``` 
