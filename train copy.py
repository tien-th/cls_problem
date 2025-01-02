import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import pandas as pd
import wandb  # Import Weights & Biases
import matplotlib.pyplot as plt
import seaborn as sns
import argparse  # For command-line arguments

from losses.Focal import FocalLoss 
from model.ResNetFM import ResNetFM as Model


# Configuration
TRAIN_INDICATOR_CSV = "split/train.csv"
VAL_INDICATOR_CSV = "split/val.csv"
num_classes = 4
batch_size = 32
epochs = 50
# learning_rate = 0.0001
learning_rate = 3e-5
momentum = 0.9
weight_decay = 1e-4
run_name = "resnetFM_focal_test_env"  # Name of the W&B run

# ------------------------------
# 1. Define Dataset and Utility Functions
# ------------------------------

# Data augmentation and preprocessing
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load dataset from CSV
def load_dataset(indicator_csv):
    # Read indicator CSV 
    indicator_df = pd.read_csv(indicator_csv)
    data  = indicator_df['image_path'].values
    labels = indicator_df['label'].values
    return data, labels

# ------------------------------
# 2. Define Training and Validation Functions
# ------------------------------

def train(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    loss = running_loss / len(train_loader.dataset)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')  # Compute F1 score
    return loss, accuracy, f1

def validate(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model.infer(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')  # Compute F1 score
    return loss, accuracy, f1, all_labels, all_preds

# ------------------------------
# 3. Checkpointing Functions
# ------------------------------

def save_checkpoint(state, path='./', run_id=None, filename="best_ckpt.pth",):
    if run_id: 
        folder = os.path.join(path, run_id)
    else:
        folder = path 
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, filename)
    torch.save(state, filename)
    wandb.save(filename)  # Upload the checkpoint to W&B

def load_checkpoint(filename, model, optimizer, scheduler):
    if not os.path.isfile(filename):
        print(f"No checkpoint found at '{filename}'")
        return model, optimizer, scheduler, 0, 0.0

    print(f"Loading checkpoint '{filename}'")
    checkpoint = torch.load(filename, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_f1 = checkpoint['best_f1']

    print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    return model, optimizer, scheduler, start_epoch, best_f1

# ------------------------------
# 4. Main Function with W&B and Resume Capability
# ------------------------------

def main():
    # ------------------------------
    # 4.1. Parse Command-Line Arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description="PyTorch Training with W&B and Resume Capability")
    parser.add_argument('--resume', type=str, default=None, help="Path to the checkpoint to resume from")
    args = parser.parse_args()

    # ------------------------------
    # 4.3. Device Configuration
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # 4.4. Load Datasets
    # ------------------------------
    train_data, train_labels = load_dataset(TRAIN_INDICATOR_CSV)
    val_data, val_labels = load_dataset(VAL_INDICATOR_CSV)

    train_dataset = CustomDataset(train_data, train_labels, train_transforms)
    val_dataset = CustomDataset(val_data, val_labels, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ------------------------------
    # 4.5. Model, Criterion, Optimizer, Scheduler Setup
    # ------------------------------
    model = Model(num_classes=num_classes)
    model = model.to(device)
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # ------------------------------
    # 4.2. Initialize W&B
    # ------------------------------
    wandb.login(key='9ab49432fdba1dc80b8e9b71d7faca7e8b324e3e')  # Login to W&B
    # Log metrics to W&B
    run = wandb.init(
        project="cls_cell",
        # entity="your_wandb_entity",  # Replace with your W&B username or team name
        config={
            "num_classes": num_classes,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "architecture": "ResNetFM",
            "loss_function": "FocalLoss"
        },
        name= run_name  # Name of the run
    )
    config = wandb.config
    run_id = run.id
    print(f"W&B Run ID: {run_id}")  # Print to console
    wandb.log({"Run ID": run_id})    # Log as a W&B metric


    # ------------------------------
    # 4.6. Resume from Checkpoint if Provided
    # ------------------------------
    start_epoch = 1
    best_f1 = 0.0
    if args.resume:
        model, optimizer, scheduler, start_epoch, best_f1 = load_checkpoint(args.resume, model, optimizer, scheduler)
        wandb.log({"Start Epoch": start_epoch, "Best F1": best_f1})

    # ------------------------------
    # 4.7. Watch the Model with W&B
    # ------------------------------
    wandb.watch(model, log="all", log_freq=100)

    # ------------------------------
    # 4.8. Training Loop
    # ------------------------------
    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        train_loss, train_acc, train_f1 = train(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_labels_list, val_preds_list = validate(val_loader, model, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        wandb.log({
            "epoch": epoch,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Train F1": train_f1,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc,
            "Val F1": val_f1,
            "Learning Rate": optimizer.param_groups[0]['lr']
        })


        # Update the learning rate scheduler based on validation F1 score
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        # Save the model if validation F1 improves
        if val_f1 > best_f1:
            best_f1 = val_f1
            print("Saving best model based on F1 score...")
            os.makedirs("models", exist_ok=True)
            checkpoint_path = f"best_model.pth"
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1
            }, path='runs', run_id=run_id,filename=checkpoint_path)

            # Log the confusion matrix using W&B's built-in function
            class_names = ['rubbish', 'unhealthy', 'healthy', 'bothcells']
            wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                                y_true=val_labels_list,
                                                                preds=val_preds_list,
                                                                class_names=class_names)})

    print(f"Training completed. Best F1 Score: {best_f1:.4f}")
    wandb.finish()  # Finish the W&B run

if __name__ == "__main__":
    main()