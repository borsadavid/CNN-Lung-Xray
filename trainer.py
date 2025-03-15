import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score, classification_report
import numpy as np

from helpers.save_load_model import save_checkpoint, load_checkpoint


import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, jaccard_score, classification_report
import numpy as np

from helpers.save_load_model import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, lr=0.001, checkpoint_path=None, accumulation_steps=2):
        """
        Initializes the Trainer.

        Args:
            model: The PyTorch model to train.
            train_dataloader (DataLoader): DataLoader for training data.
            eval_dataloader (DataLoader): DataLoader for evaluation data.
            lr (float): Learning rate for the optimizer.
            checkpoint_path (str): Path to save model checkpoints.
            accumulation_steps (int): Number of steps for gradient accumulation.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        torch.backends.cudnn.benchmark = True

        # Compute pos_weight for BCEWithLogitsLoss
        self.pos_weight = self.compute_pos_weight()

        # BCE Loss with pos_weight
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="mean")

        # AdamW optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Learning rate scheduler (Reduce LR on Plateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

        # Mixed precision scaler
        self.scaler = GradScaler()

        # Model checkpoint path
        self.checkpoint_path = checkpoint_path

        # Gradient accumulation
        self.accumulation_steps = accumulation_steps

    def compute_pos_weight(self):
        """
        Computes pos_weight tensor for BCEWithLogitsLoss based on class imbalance.
        """
        label_counts = np.sum([labels.numpy() for _, labels in self.train_dataloader.dataset], axis=0)
        total_samples = len(self.train_dataloader.dataset)

        # Compute pos_weight (inverse frequency balancing)
        pos_weight = (total_samples - label_counts) / (label_counts + 1e-6)  # Avoid division by zero
        pos_weight = np.clip(pos_weight, 0.1, 10)  # Prevent extreme values

        return torch.tensor(pos_weight, device=self.device, dtype=torch.float)

    def resume_from_checkpoint(self):
        """
        Load model checkpoint if available.
        """
        if self.checkpoint_path and isinstance(self.checkpoint_path, str):
            try:
                start_epoch = load_checkpoint(
                    self.checkpoint_path,
                    self.model,
                    self.optimizer
                )
                return start_epoch
            except FileNotFoundError:
                print(f"No checkpoint found at {self.checkpoint_path}, starting fresh.")
                return 0
        return 0

    def find_per_label_best_threshold(self, y_true, y_pred_logits):
        """
        Finds the best threshold per label to optimize F1-score.
        """
        best_thresholds = []
        for label in range(y_true.shape[1]):
            best_t = 0.5
            best_f1 = 0
            for t in np.arange(0.05, 0.9, 0.05):
                y_pred = (y_pred_logits[:, label] > t).astype(int)
                f1 = f1_score(y_true[:, label], y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            best_thresholds.append(best_t)
        return np.array(best_thresholds)

    def train(self, num_epochs=10, start_epoch=0):
        """
        Trains the model using mixed precision and gradient accumulation.
        """
        total_epochs = start_epoch + num_epochs
        best_f1 = 0.0

        for epoch in range(start_epoch, total_epochs):
            self.model.train()
            running_loss = 0.0

            for batch_idx, (images, labels) in enumerate(self.train_dataloader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels) / self.accumulation_steps  # Normalize loss for accumulation

                self.scaler.scale(loss).backward()

                # Step optimizer only every accumulation_steps batches
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                running_loss += loss.item() * images.size(0)

                del images, labels, outputs
                torch.cuda.empty_cache()

            avg_loss = running_loss / len(self.train_dataloader.dataset)
            print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {avg_loss:.4f}")

            # Evaluate on validation set
            val_f1 = self.evaluate()

            # Adjust learning rate based on validation F1 score
            self.scheduler.step(val_f1)

            # Save checkpoint if validation F1 improves
            if val_f1 > best_f1:
                best_f1 = val_f1
                if self.checkpoint_path is not None:
                    save_checkpoint(self.model, self.optimizer, epoch + 1, self.checkpoint_path)

    def evaluate(self):
        """
        Evaluates the model with per-label threshold optimization.
        """
        self.model.eval()
        dataloader = self.eval_dataloader  # Use eval_dataloader for evaluation

        y_true, y_pred_logits = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                y_true.extend(labels.cpu().numpy())
                y_pred_logits.extend(outputs.cpu().numpy())

                del images, labels, outputs
                torch.cuda.empty_cache()

        y_true = np.array(y_true)
        y_pred_logits = np.array(y_pred_logits)

        # Compute best per-label threshold
        best_thresholds = self.find_per_label_best_threshold(y_true, y_pred_logits)
        y_pred = (y_pred_logits > best_thresholds).astype(int)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        hamming = hamming_loss(y_true, y_pred)
        jaccard = jaccard_score(y_true, y_pred, average='samples')

        print(f"Accuracy: {acc:.4f} (may be misleading)")
        print(f"Jaccard Similarity: {jaccard:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")

        print("\nPer-label metrics:")
        print(classification_report(y_true, y_pred, zero_division=0))

        return f1
