import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from dataset import XRayDataset
from helpers.labels import LABELS_MAP


# Custom transformation to crop a fixed margin from all sides.
class CropMargin(object):
    def __init__(self, margin=0.15):
        """
        Args:
            margin (float): Fraction of the image width/height to crop from each side.
                            For example, 0.15 means 15% from left/right and top/bottom.
        """
        self.margin = margin

    def __call__(self, img):
        # img is expected to be a PIL image.
        width, height = img.size
        left = int(width * self.margin)
        top = int(height * self.margin)
        right = int(width * (1 - self.margin))
        bottom = int(height * (1 - self.margin))
        return img.crop((left, top, right, bottom))


def load_data(img_dir, bbox_file, train_data, eval_data, batch_size=32, num_workers=5,
              use_weighted_sampler=True, labels_map=LABELS_MAP):
    # Define transformations for training.
    train_transform = transforms.Compose([
        CropMargin(margin=0.15),  # Crop 15% from all sides.
        transforms.RandomRotation(5),  # Slight random rotation.
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Minor brightness/contrast changes.
        transforms.RandomAutocontrast(p=0.5),  # Randomly adjust the contrast.
        transforms.Resize((224, 224)),  # Resize to model input size.
        transforms.ToTensor(),  # Convert to tensor.
    ])

    # Define transformations for evaluation (deterministic).
    eval_transform = transforms.Compose([
        CropMargin(margin=0.15),  # Crop 15% from all sides.
        transforms.Resize((224, 224)),  # Resize to model input size.
        transforms.ToTensor(),  # Convert to tensor.
    ])

    # Create datasets.
    train_dataset = XRayDataset(img_dir, bbox_file, train_data, transform=train_transform)
    eval_dataset = XRayDataset(img_dir, bbox_file, eval_data, transform=eval_transform)

    # Prepare the training DataLoader.
    if use_weighted_sampler:
        if labels_map is None:
            raise ValueError("labels_map must be provided if use_weighted_sampler is True.")

        # Compute per-class weights based on training data distribution.
        num_samples = len(train_data)
        class_weights = {}
        for label in labels_map.keys():
            positive_count = train_data[label].sum()
            class_weights[label] = num_samples / (positive_count if positive_count > 0 else 1)

        # Compute a sample weight for each sample.
        sample_weights = train_data.apply(
            lambda row: max([class_weights[label] for label in labels_map.keys() if row[label] == 1] or [1.0]),
            axis=1
        ).values.astype(np.float32)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
    else:
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
