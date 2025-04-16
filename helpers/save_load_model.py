import os
import torch

from helpers.data_loader import CropMargin
from models import get_model
from PIL import Image
import numpy as np
from torchvision import transforms

def get_checkpoint_path(model_name):
    return f"lib/model/{model_name}_checkpoint.pth"

def save_checkpoint(model, model_name, optimizer, epoch):
    """Universal checkpoint saver."""
    checkpoint_path = get_checkpoint_path(model_name)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_name': model_name
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model_name, optimizer=None):
    """Load the checkpoint for the given model name."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = get_checkpoint_path(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(checkpoint['model_name'])  # Ensure get_model is available/imported
    model.load_state_dict(checkpoint['model_state_dict'])

    start_epoch = 0
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    print(f"Loaded {model_name} checkpoint from epoch {start_epoch}")
    return model, start_epoch

def load_model_for_inference(model_name):
    """
    Loads only model weights for inference (no optimizer state).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name)
    # Use the provided checkpoint path if given; otherwise, use the default one
    checkpoint_path = get_checkpoint_path(model_name)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model
