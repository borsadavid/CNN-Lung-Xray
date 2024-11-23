import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import XRayDataset


def load_data(img_dir, bbox_file, data, batch_size=32, train_ratio=0.8, num_workers=5):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Stratified split into training and evaluation datasets
    train_data, eval_data = train_test_split(
        data,
        train_size=train_ratio,
        stratify=data['Finding_Label'],
        random_state=42
    )

    # Create datasets
    train_dataset = XRayDataset(img_dir, bbox_file, train_data, transform=transform)
    eval_dataset = XRayDataset(img_dir, bbox_file, eval_data, transform=transform)

    # Create DataLoaders with optimizations
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    return train_dataloader, eval_dataloader
