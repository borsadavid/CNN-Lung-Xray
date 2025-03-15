import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from helpers.labels import LABELS_MAP


class XRayDataset(Dataset):
    def __init__(self, img_dir, bbox_file, data, transform=None):
        """
        Args:
            img_dir (str): Directory containing the X-Ray images.
            bbox_file (str): (Optional) Path to a CSV or other file containing bounding box info if used.
            data (pd.DataFrame): DataFrame that includes columns:
                                - 'Image Index': for the image file name
                                - 'Finding Labels': a '|' delimited string of findings
            transform (callable, optional): Transform to apply to the PIL image.
        """
        self.img_dir = img_dir
        self.bbox_file = bbox_file
        self.data = data
        self.transform = transform

        # The column that contains the concatenated labels
        self.label_column = 'Finding Labels'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed image
            labels (Tensor): Multi-hot encoding of labels (dim = len(LABELS_MAP))
        """
        # ----------------------------
        # 1) Get image file path
        # ----------------------------
        img_name = self.data.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)

        # If file is missing, try next index (avoid crashing)
        if not os.path.isfile(img_path):
            return self.__getitem__((idx + 1) % len(self.data))

        # ----------------------------
        # 2) Load and transform image
        # ----------------------------
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # ----------------------------
        # 3) Parse multi-labels
        # ----------------------------
        label_str = self.data.iloc[idx][self.label_column]  # e.g. 'Pneumonia|Effusion'
        label_names = [l.strip() for l in label_str.split('|')]

        # Create a zero vector for all possible labels
        labels = torch.zeros(len(LABELS_MAP), dtype=torch.float)

        # Set 1 for each label present in this image
        for lbl in label_names:
            if lbl in LABELS_MAP:
                label_idx = LABELS_MAP[lbl]  # e.g. 'Pneumonia' -> 1
                labels[label_idx] = 1

        return image, labels
