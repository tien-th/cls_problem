import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import pandas as pd
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import numpy as np

# Configuration
TRAIN_INDICATOR_CSV = "split/train.csv"
VAL_INDICATOR_CSV = "split/val.csv"
num_classes = 4
batch_size = 32
epochs = 50
learning_rate = 0.0001
model_name = "resnet50"
log_dir = "runs/resnet50_weightedLoss"  # Directory for TensorBoard logs

os.makedirs(log_dir, exist_ok=True)  # Create the log directory
os.makedirs("models", exist_ok=True)  # Directory to save models

# Data augmentation and preprocessing
train_transforms = transforms.Compose([
    # transforms.Resize((224, 224)),  # Ensures all images are of the same size
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    # transforms.Resize((224, 224)),
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

# Calculate class weights
def calculate_class_weights(labels):
    class_counts = pd.value_counts(labels)
    class_weights = 1. / class_counts
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.tensor(class_weights, dtype=torch.float)

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
    return loss, accuracy, f1, all_labels, all_preds

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, train_labels = load_dataset(TRAIN_INDICATOR_CSV)
    val_data, val_labels = load_dataset(VAL_INDICATOR_CSV)

    # Calculate class weights
    class_weights = calculate_class_weights(train_labels).to(device)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # Create datasets
    train_dataset = CustomDataset(train_data, train_labels, train_transforms)
    val_dataset = CustomDataset(val_data, val_labels, val_transforms)

    # Create sampler for balancing
    class_sample_counts = np.bincount(train_labels)
    weights = 1. / class_sample_counts
    samples_weights = weights[train_labels]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model setup
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    # Define loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_f1 = 0.0  # Initialize best F1 score

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training
        train_loss, train_acc, train_f1 = train(train_loader, model, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc, val_f1, val_labels_list, val_preds_list = validate(val_loader, model, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Train/F1_Score', train_f1, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_acc, epoch)
        writer.add_scalar('Validation/F1_Score', val_f1, epoch)

        # Log classification report and confusion matrix to TensorBoard
        if val_f1 > best_f1:
            best_f1 = val_f1
            print("Saving best model based on F1 score...")
            torch.save(model.state_dict(), f"{log_dir}/best_model_epoch_{epoch + 1}.pth")

            # # Generate classification report
            # report = classification_report(val_labels_list, val_preds_list, target_names=[f"Class {i}" for i in range(num_classes)], output_dict=True)
            # report_df = pd.DataFrame(report).transpose()
            # writer.add_table('Validation/Classification_Report', report_df, epoch)

            # Generate confusion matrix
            cm = confusion_matrix(val_labels_list, val_preds_list)
            cm_fig = plot_confusion_matrix(cm, classes=[f"Class {i}" for i in range(num_classes)])
            writer.add_figure('Validation/Confusion_Matrix', cm_fig, epoch)

    print(f"\nBest F1 Score: {best_f1:.4f}")
    writer.close()  # Close the TensorBoard writer

# Utility function to plot confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.close()
    return plt.gcf()

if __name__ == "__main__":
    main()