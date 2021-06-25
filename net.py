from torchvision import models
from torch import nn


def fine_tune_model():
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_features, 2))
    model_ft = model_ft.cuda()
    return model_ft