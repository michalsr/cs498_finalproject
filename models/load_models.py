import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

import torch.utils.model_zoo as model_zoo
def load_model(model_type,num_classes,pretrained,from_file=False):
    fine_tune_needed = False
    if pretrained == 'ImageNet':
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50()
    if pretrained == 'moco':
        checkpoint = model_zoo.load_url(
            'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar')
        pre_dict = {str.replace(k, 'module.encoder_q.', ''): v for k, v in checkpoint['state_dict'].items()}
        state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pre_dict.items() if k in state_dict}
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)

    if model.fc.in_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features,num_classes)
        fine_tune_needed = True
    return model, fine_tune_needed


