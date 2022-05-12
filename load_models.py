import numpy as np
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def load_models(model_config, **kwargs):
    model = None
    if model_config['model_name']=='TransUnet':
        model = load_TransUnet(**model_config, **kwargs)
    else:
        raise Exception(f"{model_config['model_name']} does not exists!")

    return model
    
def load_TransUnet(model_name, vit_name, vit_patches_size, n_skip, pretrained_weights, img_shape, class_list, **kwargs):
    vit_name = vit_name
    img_size = img_shape[0]
    vit_patches_size = vit_patches_size
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = len(class_list)+1
    config_vit.n_skip = n_skip

    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
    model.load_from(weights=np.load(pretrained_weights))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded {model_name}, Total Parameters : {pytorch_total_params:,}')

    return model


