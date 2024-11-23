import os

import numpy as np
import torch
from models import get_model
from PIL import Image
from torchvision import transforms

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model_name, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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
    # Define preprocessing steps for inference (deterministic, no augmentations)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Move image to the same device as the model
    device = next(model.parameters()).device
    image = image.to(device)

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Forward pass
        outputs = model(image)

        # Apply sigmoid activation to logits to get probabilities
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()

    # Binarize predictions based on the threshold
    binary_predictions = (probabilities > threshold).astype(int)

    # Determine the label with the highest probability
    max_prob_index = np.argmax(probabilities)
    max_prob_label = list(labels_map.keys())[max_prob_index]
    max_prob_value = probabilities[max_prob_index]

    # Print predictions
    print("Predicted probabilities and labels:")
    for label, prob, pred in zip(labels_map.keys(), probabilities, binary_predictions):
        print(f"{label}: Probability = {prob:.4f}, Predicted = {'Yes' if pred else 'No'}")

    # Collect positive predictions (above threshold)
    positive_labels = [label for label, pred in zip(labels_map.keys(), binary_predictions) if pred]

    # Display results
    print("\nResults:")
    if positive_labels:
        print(f"Labels above threshold: {', '.join(positive_labels)}")
    else:
        print("No labels above threshold.")

    # Display the label with the highest probability
    print(f"Label with the highest probability: {max_prob_label} (Probability = {max_prob_value:.4f})")

