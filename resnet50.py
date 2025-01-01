import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import pandas as pd
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard

# Configuration
TRAIN_INDICATOR_CSV = "split/train.csv"
VAL_INDICATOR_CSV = "split/val.csv"
num_classes = 4
batch_size = 32
epochs = 50
learning_rate = 0.0001
k_folds = 5
# model_name = "vit_small_patch32_224.augreg_in21k_ft_in1k"
model_name = "resnet50"
# log_dir = "runs/vit"  # Directory for TensorBoard logs
log_dir = "runs/resnet50"  # Directory for TensorBoard logs

os.makedirs(log_dir, exist_ok=True)  # Create the log directory

# Data augmentation and preprocessing
train_transforms = transforms.Compose([
    # transforms.Resize((224, 224)),  # Uncommented for proper image sizing
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset
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

# Load dataset
def load_dataset(indicator_csv):
    # Read indicator CSV 
    indicator_df = pd.read_csv(indicator_csv)
    data  = indicator_df['image_path'].values
    labels = indicator_df['label'].values
    return data, labels

# Training function
def train(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader, desc="Training"):
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

# Validation function
def validate(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
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
    return loss, accuracy, f1

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, train_labels = load_dataset(TRAIN_INDICATOR_CSV)
    val_data, val_labels = load_dataset(VAL_INDICATOR_CSV)

    best_f1 = 0.0  # Initialize best F1 score

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
    #     print(f"\nFold {fold + 1}/{k_folds}")

        # Split data

    train_dataset = CustomDataset(train_data, train_labels, train_transforms)
    val_dataset = CustomDataset(val_data, val_labels, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training and validation
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc, train_f1 = train(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(val_loader, model, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Log metrics to TensorBoard
        global_step = epoch
        writer.add_scalar(f'Train_Loss', train_loss, global_step)
        writer.add_scalar(f'Train_Accuracy', train_acc, global_step)
        writer.add_scalar(f'Train_F1', train_f1, global_step)
        writer.add_scalar(f'Val_Loss', val_loss, global_step)
        writer.add_scalar(f'Val_Accuracy', val_acc, global_step)
        writer.add_scalar(f'Val_F1', val_f1, global_step)

        # Save the model if validation F1 improves
        if val_f1 > best_f1:
            best_f1 = val_f1
            print("Saving best model based on F1 score...")
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"{log_dir}/best_model_{global_step + 1}.pth")

    print(f"Best F1 Score: {best_f1:.4f}")
    writer.close()  # Close the TensorBoard writer

if __name__ == "__main__":
    main()