"""
Training script for the Cell CLASSIFIER model.
Located at: src/train/train_classifier.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm 
import os
from pathlib import Path

from src.data.dataset import make_loaders


DATA_ROOT = "data" 
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
MODEL_SAVE_DIR = "models"
MODEL_SAVE_NAME = "cell_classifier_best.pth"

def calculate_accuracy(y_pred, y_true):
    """Calculates accuracy for binary classification"""

    predicted = (torch.sigmoid(y_pred) > 0.5).float()
    correct = (predicted == y_true).float()
    return correct.sum() / len(correct)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() 
    
    running_loss = 0.0
    running_acc = 0.0
    
    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_acc += calculate_accuracy(outputs, labels) * images.size(0)
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    
    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    running_acc = 0.0
    

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_acc += calculate_accuracy(outputs, labels) * images.size(0)
            
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    
    return epoch_loss, epoch_acc


def main():
    Path(MODEL_SAVE_DIR).mkdir(exist_ok=True)
    save_path = Path(MODEL_SAVE_DIR) / MODEL_SAVE_NAME

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    train_loader, val_loader, _ = make_loaders(
        data_root=DATA_ROOT, 
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count() // 2 
    )
    if train_loader is None:
        print("Failed to create data loaders. Exiting.")
        return
    print("Data loaded successfully.")

    print("Loading pre-trained ResNet18 model...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1) 

    model.to(device)
 
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with accuracy: {best_val_acc:.4f} at {save_path}")
            
    print("\n--- Training Complete ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {save_path}")

if __name__ == "__main__":
    main()