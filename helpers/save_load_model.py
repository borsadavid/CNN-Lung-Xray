import os
import torch

from helpers.data_loader import CropMargin
from models import get_model
from PIL import Image
import numpy as np
from torchvision import transforms

def save_checkpoint(model, optimizer, epoch, save_path):
    """
    Save model and optimizer states to continue training later.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model (and optionally optimizer) states from a checkpoint.

    Args:
        checkpoint_path (str): path to the .pth file
        model (nn.Module): model instance with the same architecture
        optimizer (torch.optim.Optimizer, optional): optimizer instance

    Returns:
        int: epoch to resume from
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    start_epoch = 0
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    print(f"Loaded checkpoint from {checkpoint_path}, epoch={start_epoch}")
    return start_epoch

def load_model_for_inference(model_name, checkpoint_path):
    """
    Loads only model weights for inference (no optimizer state).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, labels_map, threshold=0.5):
    """
    Predict and print the labels for a single image using the trained model.
    Mimics the evaluation logic in the Trainer class for consistency.

    Args:
        model: Loaded PyTorch model.
        image_path (str): Path to the X-ray image.
        labels_map (dict): Mapping of labels to their indices.
        threshold (float): Threshold for binary classification.
    """
    preprocess = transforms.Compose([
        CropMargin(margin=0.15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  # Add batch dimension

    device = next(model.parameters()).device
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

    # Binarize predictions
    binary_predictions = (probabilities > threshold).astype(int)

    # Determine the label with the highest probability
    max_prob_index = np.argmax(probabilities)
    max_prob_label = list(labels_map.keys())[max_prob_index]
    max_prob_value = probabilities[max_prob_index]

    # Print predictions
    print("Predicted probabilities and labels:")
    for label, prob, pred in zip(labels_map.keys(), probabilities, binary_predictions):
        print(f"{label}: Probability = {prob:.4f}, Predicted = {'Yes' if pred else 'No'}")

    # Collect positive predictions
    positive_labels = [label for label, pred in zip(labels_map.keys(), binary_predictions) if pred]

    print("\nResults:")
    if positive_labels:
        print(f"Labels above threshold: {', '.join(positive_labels)}")
    else:
        print("No labels above threshold.")

    print(f"Label with the highest probability: {max_prob_label} (Probability = {max_prob_value:.4f})")
