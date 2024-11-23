import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
import numpy as np

class Trainer:
    def __init__(self, model, dataloader, lr=0.001):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Enable cuDNN benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Mixed precision scaler
        self.scaler = GradScaler()

    def train(self, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0

            for images, labels in self.dataloader:
                # Move data to GPU
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Mixed precision forward pass
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Accumulate loss
                running_loss += loss.item() * images.size(0)

                # Free memory
                del images, labels, outputs
                torch.cuda.empty_cache()

            # Log average loss per epoch
            avg_loss = running_loss / len(self.dataloader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, threshold):
        self.model.eval()
        dataloader = self.dataloader

        y_true, y_pred_logits = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                # Move data to GPU
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Move outputs to CPU and clear GPU memory
                y_true.extend(labels.cpu().numpy())
                y_pred_logits.extend(outputs.cpu().numpy())

                # Free memory
                del images, labels, outputs
                torch.cuda.empty_cache()

        # Convert predictions and true labels to numpy arrays
        y_true = np.array(y_true)
        y_pred_logits = np.array(y_pred_logits)

        # Binarize predictions with the new logic:
        # If no label surpasses the threshold, pick the label with the highest value
        y_pred = np.zeros_like(y_pred_logits, dtype=int)
        for i, row in enumerate(y_pred_logits):
            above_threshold = row > threshold
            if np.any(above_threshold):
                y_pred[i] = above_threshold.astype(int)
            else:
                # Select the label with the highest value
                max_index = np.argmax(row)
                y_pred[i, max_index] = 1

        # Compute evaluation metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        hamming = hamming_loss(y_true, y_pred)

        # Print evaluation metrics
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")

