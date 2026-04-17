"""
Improved Model Training Script - Fixes Overfitting Issues
Train a CNN model for deepfake detection with better generalization

Key improvements:
- Aggressive data augmentation
- Dropout layers for regularization
- Unfroze more layers
- Class balancing
- Better hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms, models
import os
from pathlib import Path
import argparse
import random
import numpy as np
from PIL import Image
import json

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class DeepfakeDetectorV2(nn.Module):
    """Improved model with dropout and better layer unfreezing"""
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(DeepfakeDetectorV2, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        
        # Only freeze layer1 (more aggressive unfreezing)
        for param in self.resnet.layer1.parameters():
            param.requires_grad = False
        
        # Add dropout before classification
        num_features = self.resnet.fc.in_features
        self.dropout = nn.Dropout(dropout_rate)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

class DeepfakeDataset(Dataset):
    """Custom dataset for loading images from real/fake folders"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load real images (label 0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img_file in os.listdir(real_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    self.images.append(os.path.join(real_dir, img_file))
                    self.labels.append(0)
        
        # Load fake images (label 1)
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            for img_file in os.listdir(fake_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    self.images.append(os.path.join(fake_dir, img_file))
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading image {self.images[idx]}: {e}")
            return torch.zeros(3, 224, 224), self.labels[idx]

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Train improved deepfake detector model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (reduced for better generalization)')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate (reduced)')
    parser.add_argument('--data-dir', type=str, default='data', help='Path to training data')
    parser.add_argument('--model-path', type=str, default='models/deepfake_detector_v2.pth', help='Path to save model')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # AGGRESSIVE Data augmentation to improve generalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        
        # Geometric transformations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        
        # Color/brightness augmentation
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        
        # Gaussian blur
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # Random perspective
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        
        # Convert to tensor and normalize
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation transform (minimal)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        print("Please organize your data as:")
        print("  data/")
        print("    real/")
        print("      image1.jpg")
        print("    fake/")
        print("      image2.jpg")
        return
    
    # Create datasets
    print("Loading dataset...")
    train_dataset = DeepfakeDataset(args.data_dir, transform=train_transform)
    
    if len(train_dataset) == 0:
        print("Error: No images found in dataset!")
        return
    
    print(f"Found {len(train_dataset)} images")
    real_count = sum(1 for l in train_dataset.labels if l == 0)
    fake_count = sum(1 for l in train_dataset.labels if l == 1)
    print(f"  Real: {real_count}")
    print(f"  Fake: {fake_count}")
    
    # Split into train and validation (80-20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Class balancing with WeightedRandomSampler
    train_labels = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
    class_counts = [sum(1 for l in train_labels if l == c) for c in range(2)]
    class_weights = [len(train_labels) / (2 * c) for c in class_counts]
    sample_weights = [class_weights[train_labels[i]] for i in range(len(train_labels))]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Create data loaders
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
    
    # Model with dropout
    model = DeepfakeDetectorV2(num_classes=2, dropout_rate=0.5).to(device)
    
    # Loss with class weighting
    class_weights_tensor = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    
    # Reduce learning rate when validation accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    print(f"\n✓ Improved Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} (smaller = better generalization)")
    print(f"  Learning rate: {args.lr} (reduced)")
    print(f"  Dropout: 0.5 (prevents overfitting)")
    print(f"  Data augmentation: AGGRESSIVE")
    print(f"  Class balancing: ENABLED")
    print(f"  Weight decay: 0.0001 (L2 regularization)")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print("-" * 54)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.2f} {val_loss:<12.4f} {val_acc:<12.2f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_path)
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 54)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.model_path}")
    
    # Save history
    history_path = args.model_path.replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

if __name__ == '__main__':
    main()
