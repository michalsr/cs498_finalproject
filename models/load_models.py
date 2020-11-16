import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

def load_model(model_type,num_classes,pretrained=True,from_file=False):
    fine_tune_needed = False
    model = models.resnet50(pretrained=pretrained)
    if model.fc.in_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features,num_classes)
        fine_tune_needed = True
    return model, fine_tune_needed


