import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from model import DrivingCNN
import torch.nn as nn
import torch.optim as optim
import albumentations as A  # pip install albumentations

class DrivingDataset(Dataset):
    def __init__(self, csv_file, img_dir, augment=False):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.augment = augment
        
        # Data augmentation for training
        if augment:
            self.aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.GaussNoise(p=0.2),  # Fixed: removed var_limit
                A.HorizontalFlip(p=0.5),  # Flip image AND steering
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = cv2.imread(os.path.join(self.img_dir, row["image"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        steering = row["steering"]
        throttle = row["throttle"]
        
        # Apply augmentation
        if self.augment:
            augmented = self.aug(image=img)
            img = augmented["image"]
            
            # If horizontally flipped, invert steering
            if augmented.get("replay", {}).get("transforms", []):
                for t in augmented["replay"]["transforms"]:
                    if t.get("__class_fullname__") == "HorizontalFlip" and t.get("applied"):
                        steering = -steering
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        y = np.array([steering, throttle], dtype=np.float32)
        return torch.tensor(img), torch.tensor(y)

# Load dataset
full_dataset = DrivingDataset("dataset/labels.csv", "dataset/images", augment=True)

# Split
train_idx, val_idx = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
train_set = Subset(full_dataset, train_idx)

# Validation WITHOUT augmentation
val_dataset = DrivingDataset("dataset/labels.csv", "dataset/images", augment=False)
val_set = Subset(val_dataset, val_idx)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DrivingCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Early stopping
patience = 7
best_val_loss = float('inf')
epochs_no_improve = 0

num_epochs = 100

print(f"\nTraining samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}\n")

for epoch in range(num_epochs):
    # Training
    model.train()
    total_train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | LR: {current_lr:.6f}")

    # Update learning rate
    old_lr = current_lr
    scheduler.step(avg_val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr < old_lr:
        print(f"  → Learning rate reduced to {new_lr:.6f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "model_best.pth")
        print(f"  → New best model saved!")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print(f"\nBest validation loss: {best_val_loss:.5f}")
torch.save(model.state_dict(), "model_final.pth")
