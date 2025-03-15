import os
from helpers.data_loader import load_data
from helpers.labels import LABELS_MAP
from helpers.balance_data import preprocess_data, balance_and_upsample_data
from helpers.save_load_model import save_checkpoint, load_model_for_inference, predict_image
from models import get_model
from trainer import Trainer

# Define constants for paths and parameters
IMG_DIR = 'lib/training/images_1/'
EVALUATION_IMG_DIR = 'lib/evaluation/images/'
BBOX_FILE = 'lib/training/bounding_box_data.csv'
DATA_ENTRY_FILE = 'lib/training/data_entry.csv'
checkpoint_path = "lib/model/model_checkpoint.pth"


def setup_data_and_dataloaders(data_file, img_dir, bbox_file, labels_map, data_entries, train_ratio):
    print("Preprocessing data (multi-label, no single-label filtering)...")

    # Load and preprocess the data
    data = preprocess_data(data_file, labels_map, data_entries)
    print(f"Total data rows available: {len(data)}")

    # Split the data into training and evaluation sets.
    num_train = int(train_ratio * len(data))
    train_data = data.sample(n=num_train, random_state=42)
    eval_data = data.drop(train_data.index)

    # Shuffle the splits to avoid ordering issues.
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    eval_data = eval_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Apply balancing and upsampling ONLY to the training data.
    print("Balancing and upsampling training data...")
    train_data = balance_and_upsample_data(train_data, labels_map)

    print("Creating train & eval DataLoaders...")
    train_dataloader, eval_dataloader = load_data(
        img_dir, bbox_file, train_data, eval_data
    )
    return train_dataloader, eval_dataloader

def train_and_save_model(model_name, train_dataloader, eval_dataloader, num_epochs, checkpoint_path, lr=0.001, accumulation_steps=2):
    """
    Trains the model and saves checkpoints.

    Args:
        model_name (str): Name of the model architecture.
        train_dataloader (DataLoader): DataLoader for training data.
        eval_dataloader (DataLoader): DataLoader for evaluation data.
        num_epochs (int): Number of epochs to train.
        checkpoint_path (str): Path to save model checkpoints.
        lr (float): Learning rate for the optimizer.
        accumulation_steps (int): Number of steps for gradient accumulation.
    """
    # Create model
    model = get_model(model_name)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        lr=lr,
        checkpoint_path=checkpoint_path,
        accumulation_steps=accumulation_steps
    )

    # Optionally resume from checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print("Resuming from existing checkpoint...")
        start_epoch = trainer.resume_from_checkpoint()
        trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)
    else:
        print("Starting training from scratch...")
        trainer.train(num_epochs=num_epochs, start_epoch=0)

    print("Training complete.")
    return trainer.model

def evaluate_model_on_dataloader(model, train_dataloader, eval_dataloader):
    print("Evaluating model...")
    trainer = Trainer(model, train_dataloader, eval_dataloader)
    trainer.evaluate()

def predict_and_display(model, image_path, labels_map):
    predict_image(model, image_path, labels_map)

if __name__ == "__main__":
    train_ratio = 0.8
    model_name = "efficientnet_b0"
    num_epochs = 3
    data_entries = 45000

    # Step 1: Setup data
    train_dataloader, eval_dataloader = setup_data_and_dataloaders(
        DATA_ENTRY_FILE, IMG_DIR, BBOX_FILE, LABELS_MAP, data_entries, train_ratio
    )

    # Step 2: Train (or resume) and get final model
    model = train_and_save_model(model_name, train_dataloader, eval_dataloader, num_epochs, checkpoint_path)

    #model = load_model_for_inference(model_name, checkpoint_path)
    # Step 3: Evaluate
    #evaluate_model_on_dataloader(model, train_dataloader, eval_dataloader)

    # Or load and predict:

    #predict_and_display(model, EVALUATION_IMG_DIR + "00020968_000.png", LABELS_MAP)

# Epoch [1/10], Loss: 0.3890
# Epoch [2/10], Loss: 0.2266
# Epoch [3/10], Loss: 0.1776
# Epoch [4/10], Loss: 0.1460
# Epoch [5/10], Loss: 0.1096
# Epoch [6/10], Loss: 0.0901
# Epoch [7/10], Loss: 0.0718
# Epoch [8/10], Loss: 0.0602
# Epoch [9/10], Loss: 0.0488
# Epoch [10/10], Loss: 0.0400

# Evaluation on data_entry.csv:
#Resnet18
# Accuracy: 0.6814
# Precision: 0.9026
# Recall: 0.6917
# F1 Score: 0.7832
# Hamming Loss: 0.0319

#Resnet50
# Accuracy: 0.7108
# Precision: 0.9143
# Recall: 0.7206
# F1 Score: 0.8060
# Hamming Loss: 0.0289

#Resnet50 with condition if threshold is not met by any, return the highest label value
# Accuracy: 0.7925
# Precision: 0.7940
# Recall: 0.8039
# F1 Score: 0.7989
# Hamming Loss: 0.0337

#Resnet50 with all above but 45000 entries instead of 11000 (0.5 threshold)
# Accuracy: 0.8153
# Precision: 0.8160
# Recall: 0.8235
# F1 Score: 0.8197
# Hamming Loss: 0.0302

#Efficientnet after multiple scaling/weighting improvements
# Accuracy: 0.9051 (may be misleading)
# Jaccard Similarity: 0.9186
# Precision: 0.9324
# Recall: 0.9354
# F1 Score: 0.9339
# Hamming Loss: 0.0131
#
# Per-label metrics:
#               precision    recall  f1-score   support
#
#            0       0.93      0.94      0.93       814
#            1       0.91      0.95      0.93       369
#            2       0.90      0.89      0.89      1266
#            3       0.91      0.97      0.94       342
#            4       0.94      0.98      0.96       139
#            5       0.93      0.99      0.96       166
#            6       0.92      0.88      0.90       914
#            7       0.87      0.97      0.92       228
#            8       0.87      0.94      0.91       251
#            9       0.97      0.99      0.98       389
#           10       0.95      0.89      0.92       501
#           11       0.95      0.95      0.95      5184
#
#    micro avg       0.93      0.94      0.93     10563
#    macro avg       0.92      0.94      0.93     10563
# weighted avg       0.93      0.94      0.93     10563
#  samples avg       0.92      0.93      0.92     10563

#Challanges:
#Some samples appeared more often and some were rare, had to balance them
#Finding the right model
#Optimizing with processor cores and using GPU batching
#Quantity of images
#Threshold comparison or picking the label with the max probability




