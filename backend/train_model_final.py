"""
TruthLens AI - Final Training Script (Windows Compatible)
Proper training with validation and best practices
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
from PIL import Image
from pathlib import Path
import json
import argparse
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {DEVICE}")

# ============================================================
# DATASET CLASS
# ============================================================
class DeepfakeDataset(Dataset):
    """Load images from real/fake directories"""
    
    def __init__(self, data_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        data_path = Path(data_dir)
        
        # Load real images (label 0)
        real_dir = data_path / "real"
        if real_dir.exists():
            for img_file in real_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    self.images.append(str(img_file))
                    self.labels.append(0)
        
        # Load fake images (label 1)
        fake_dir = data_path / "fake"
        if fake_dir.exists():
            for img_file in fake_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    self.images.append(str(img_file))
                    self.labels.append(1)
        
        print(f"📊 Loaded {len(self.images)} images")
        print(f"   Real: {sum(1 for l in self.labels if l == 0)}")
        print(f"   Fake: {sum(1 for l in self.labels if l == 1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception as e:
            print(f"⚠️  Error loading {self.images[idx]}: {e}")
            return torch.zeros(3, 224, 224), self.labels[idx]

# ============================================================
# MODEL
# ============================================================
class DeepfakeDetectorModel(nn.Module):
    """ResNet50 with dropout for better generalization"""
    
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze only early layers
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        
        # Replace final layer with dropout + classification head
        in_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Train Deepfake Detector')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--model-path', type=str, default='models/deepfake_detector.pth', help='Model save path')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🚀 DEEPFAKE DETECTOR TRAINING")
    print("=" * 60)
    
    # ============================================================
    # DATA LOADING
    # ============================================================
    print("\n📂 Loading dataset...")
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        
        # Geometric
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        
        # Color
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        
        # Normalize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    dataset = DeepfakeDataset(args.data_dir, transform=train_transform)
    
    if len(dataset) == 0:
        print("❌ ERROR: No images found!")
        print(f"   Check that {args.data_dir}/real/ and {args.data_dir}/fake/ exist")
        return
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Class balancing
    train_labels = [dataset.labels[i] for i in train_dataset.indices]
    class_counts = [sum(1 for l in train_labels if l == c) for c in range(2)]
    class_weights = [len(train_labels) / (2 * c) for c in class_counts]
    sample_weights = [class_weights[train_labels[i]] for i in range(len(train_labels))]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # ============================================================
    # MODEL & TRAINING SETUP
    # ============================================================
    print("🧠 Creating model...")
    model = DeepfakeDetectorModel(dropout_rate=0.5).to(DEVICE)
    
    # Loss with class weighting
    class_weights_tensor = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    print("\n" + "=" * 60)
    print("⚙️  TRAINING CONFIG")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Train Samples: {len(train_dataset)}")
    print(f"Val Samples: {len(val_dataset)}")
    print(f"Dropout: 0.5")
    print(f"Data Augmentation: ENABLED")
    print(f"Class Balancing: ENABLED")
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    print("\n" + "=" * 60)
    print("🔥 TRAINING...")
    print("=" * 60)
    
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Schedule
        scheduler.step(val_acc)
        
        # Save history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_path)
            print(f"   ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 10:
            print(f"\n⏹️  Early stopping (no improvement for 10 epochs)")
            break
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.model_path}")
    
    # Save history
    history_path = args.model_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to: {history_path}")
    
    # Print final stats
    print(f"\n📈 Final Results:")
    print(f"   Epoch {history['epochs'][-1]}")
    print(f"   Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"   Val Accuracy: {history['val_acc'][-1]:.2f}%")
    
    # Check for overfitting
    gap = history['train_acc'][-1] - history['val_acc'][-1]
    if gap > 15:
        print(f"\n⚠️  WARNING: Large gap between train ({history['train_acc'][-1]:.2f}%) and val ({history['val_acc'][-1]:.2f}%) accuracy!")
        print("   This indicates overfitting. Consider:")
        print("   - Adding more training data")
        print("   - Using stronger data augmentation")
        print("   - Reducing model complexity")
    else:
        print(f"\n✅ Good generalization (gap: {gap:.2f}%)")

if __name__ == '__main__':
    main()
