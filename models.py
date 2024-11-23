import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from helpers.labels import LABELS_MAP

def get_model(model_name, num_classes=len(LABELS_MAP)):
    """
    Returns a specified pretrained model with the last layer modified for multi-label classification.
    Args:
        model_name (str): Name of the model. Supported: resnet18, resnet50, mobilenet_v2, efficientnet_b0, shufflenet_v2_x1_0.
        num_classes (int): Number of output classes.
    Returns:
        torch.nn.Module: Pretrained model with modified classification head.
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes)
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported")

    return model
