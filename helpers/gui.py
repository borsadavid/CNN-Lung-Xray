import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import torch
import numpy as np
from torchvision import transforms

from helpers.data_loader import CropMargin
from helpers.save_load_model import load_model_for_inference
from helpers.labels import LABELS_MAP
from tkinterdnd2 import *

def predict_image(model, image_path, labels_map, threshold=0.01):

    preprocess = transforms.Compose([
        CropMargin(margin=0.15),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)

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

    # Collect output
    output = "Predicted probabilities and labels:\n"
    for label, prob, pred in zip(labels_map.keys(), probabilities, binary_predictions):
        output += f"{label}: Probability = {prob:.4f}, Predicted = {'Yes' if pred else 'No'}\n"

    # Collect positive predictions
    positive_labels = [label for label, pred in zip(labels_map.keys(), binary_predictions) if pred]

    output += "\nResults:\n"
    if positive_labels:
        output += f"Labels above threshold: {', '.join(positive_labels)}\n"
    else:
        output += "No labels above threshold.\n"

    output += f"Label with the highest probability: {max_prob_label} (Probability = {max_prob_value:.4f})\n"

    return output


def launch_gui():
    """
    Create a simple GUI with a model selection dropdown, drag-and-drop image area,
    and a text area to display prediction results.
    """
    root = TkinterDnD.Tk()
    root.title("Lung X-ray Prediction")
    root.geometry("600x600")

    # Model selection
    model_label = tk.Label(root, text="Select Model:")
    model_label.pack(pady=5)

    model_options = ['mobilenet_v2', 'resnet50', 'efficientnet_b0']
    selected_model = tk.StringVar(value=model_options[0])
    model_dropdown = ttk.Combobox(root, textvariable=selected_model, values=model_options, state='readonly')
    model_dropdown.pack(pady=5)

    # Image display area
    image_label = tk.Label(root, text="Drag and drop or click to select an image", relief="solid", width=50, height=15)
    image_label.pack(pady=10)

    # Text area for results
    result_text = tk.Text(root, height=15, width=60)
    result_text.pack(pady=10)

    # Load initial model
    current_model = [load_model_for_inference(selected_model.get())]

    def load_selected_model(event):
        model_name = selected_model.get()
        try:
            current_model[0] = load_model_for_inference(model_name)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Loaded model: {model_name}\n")
        except Exception as e:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Error loading model: {str(e)}\n")

    model_dropdown.bind('<<ComboboxSelected>>', load_selected_model)

    def handle_image_drop(event):
        if event.data:
            # Handle drag-and-drop image
            image_path = event.data.strip('{}')  # Clean up tkinter drag-and-drop format
            display_and_predict(image_path)

    def select_image():
        image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if image_path:
            display_and_predict(image_path)

    def display_and_predict(image_path):
        try:
            # Display image
            img = Image.open(image_path)
            img = img.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            image_label.configure(image=photo, text="")
            image_label.image = photo  # Keep a reference
            # Predict
            output = predict_image(current_model[0], image_path, LABELS_MAP)
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, output)
        except Exception as e:
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, f"Error loading image: {str(e)}\n")

    # Enable drag-and-drop
    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', handle_image_drop)

    # Enable click to select
    image_label.bind("<Button-1>", lambda e: select_image())

    root.mainloop()