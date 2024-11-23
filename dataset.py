import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from helpers.labels import LABELS_MAP

# Define a custom dataset class that inherits from PyTorch's Dataset class
class XRayDataset(Dataset):
    def __init__(self, img_dir, bbox_file, data, transform=None):
        self.img_dir = img_dir  # Directory containing the images
        self.transform = transform  # Transformation to be applied to the images (if any)
        self.data = data
        self.label_column = 'Finding Labels'  # Column containing disease labels

    def __len__(self):
        # Return the total number of data samples (based on the selected data file)
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image file name using the data index
        img_name = self.data.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)  # Create the full path to the image

        # Check if the image file exists, if not get the next available image
        if not os.path.isfile(img_path):
            return self.__getitem__((idx + 1) % len(self.data))

        # Open the image and convert it to RGB format
        image = Image.open(img_path).convert("RGB")

        # Apply the transformation if specified
        if self.transform:
            image = self.transform(image)

        # Get the disease labels and map them to their corresponding integers
        label_name = self.data.iloc[idx][self.label_column]
        label_names = label_name.split('|')  # Split labels by '|' as there can be multiple diseases on the same image
        valid_labels = []

        # Multi-hot encode the labels
        labels = torch.zeros(len(LABELS_MAP))  # Create a zero vector for all possible labels
        for l in label_names:
            label_idx = LABELS_MAP.get(l.strip(), None)
            if label_idx is not None:
                labels[label_idx] = 1  # Set the corresponding label index to 1
                valid_labels.append(l.strip())

        # Return the (image, labels) pair
        return image, labels


# Detailed explanation:
# 1. The XRayDataset class is a custom PyTorch dataset that handles X-ray images, bounding box data, and labels.
# 2. The constructor (__init__) takes the image directory, bounding box file, data entry file, transformations, and a flag `use_data_entry`.
#    - It loads either the bounding box or the data entry CSV file based on the flag.
#    - It also sets up a mapping for each disease label to a unique integer.
# 3. The __len__ method returns the total number of images in the dataset.
# 4. The __getitem__ method is used to access individual samples from the dataset by index.
#    - It first constructs the image path and checks if the file exists.
#    - If the file doesn't exist, it tries the next image to avoid crashes.
#    - It then opens the image, applies transformations, and retrieves the corresponding disease label.
#    - If multiple disease labels exist, it takes the first label.
#    - Finally, it returns the processed image and label.

