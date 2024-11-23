from helpers.data_loader import load_data
from helpers.labels import LABELS_MAP
from helpers.balance_data import preprocess_and_balance_data
from helpers.save_load_model import save_model, load_model, predict_image
from models import get_model
from trainer import Trainer

# Define constants for paths and parameters
IMG_DIR = 'lib/training/images_1/'
BBOX_FILE = 'lib/training/bounding_box_data.csv'
DATA_ENTRY_FILE = 'lib/training/data_entry.csv'
checkpoint_path = "lib/model/model_checkpoint.pth"


def setup_data_and_dataloaders(data_file, img_dir, bbox_file, labels_map, data_entries, train_ratio):
    """
    Preprocesses data and prepares the training and evaluation dataloaders.
    """
    print("Preprocessing and balancing data...")
    balanced_data = preprocess_and_balance_data(data_file, labels_map, data_entries)
    print(f"Total data rows for training: {len(balanced_data)}")

    print("Splitting data into training and evaluation datasets...")
    train_dataloader, eval_dataloader = load_data(img_dir, bbox_file, balanced_data, train_ratio=train_ratio)
    return train_dataloader, eval_dataloader


def train_and_save_model(model_name, train_dataloader, num_epochs):
    """
    Trains the model and saves it to a file.
    """
    print(f"Initializing and training model: {model_name}")
    model = get_model(model_name)
    trainer = Trainer(model, train_dataloader)
    trainer.train(num_epochs=num_epochs)

    print(f"Saving trained model to {checkpoint_path}...")
    save_model(model, checkpoint_path)
    return model


def evaluate_model_on_dataloader(model, eval_dataloader, threshold):
    """
    Evaluates the model on the provided evaluation dataloader.
    """
    print("Evaluating model...")
    trainer = Trainer(model, eval_dataloader)
    trainer.evaluate(threshold)
    return


def predict_and_display(model, image_path, labels_map):
    """
    Performs prediction on a single image and displays the result.
    """
    print(f"Predicting on image: {image_path}")
    predict_image(model, image_path, labels_map)

if __name__ == "__main__":
    # Essential variables
    train_ratio = 0.8
    model_name = "resnet50"
    num_epochs = 15
    data_entries = 45000

    # Setup data and dataloaders
    train_dataloader, eval_dataloader = setup_data_and_dataloaders(
        DATA_ENTRY_FILE, IMG_DIR, BBOX_FILE, LABELS_MAP, data_entries, train_ratio
    )

    # Train and save the model OR load an existing model
    #model = train_and_save_model(model_name, train_dataloader, num_epochs)
    model = load_model(model_name, checkpoint_path)

    # Evaluate the model
    evaluate_model_on_dataloader(model, eval_dataloader, 0.96)

    # Make predictions
    #predict_and_display(model, IMG_DIR + "00000076_000.png", LABELS_MAP)

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

#Challanges:
#Some samples appeared more often and some were rare, had to balance them
#Finding the right model
#Optimizing with processor cores and using GPU batching
#Quantity of images
#Threshold comparison or picking the label with the max probability