# src/train.py
"""
Model eğitim scripti
"""

import torch
import os
import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from src.data.dataset import UrineFieldDataset
from src.models.cnn_model import SimpleFineTune
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import json


def train_epoch(model, loader, criterion, optimizer, device):
    """Bir epoch eğitim"""
    model.train()
    total = 0
    correct = 0
    loss_sum = 0
    
    for x, y in tqdm(loader, desc="Training"):
        x = x.to(device)
        y = y.to(device)
        
        # Forward
        logits = model(x)
        loss = criterion(logits, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrikler
        loss_sum += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return loss_sum / len(loader), correct / total


def validate(model, loader, criterion, device):
    """Validation"""
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation"):
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return loss_sum / len(loader), correct / total


def train(args):
    """Ana eğitim fonksiyonu"""
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Dataset
    full_dataset = UrineFieldDataset(args.data, train=True)
    
    # Train/Val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    model = SimpleFineTune(
        n_classes=args.n_classes, 
        pretrained=args.pretrained,
        backbone=args.backbone
    ).to(device)
    
    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Eğitim döngüsü
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # History
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Best model kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history
            }, args.out)
            print(f"✓ Best model saved (val_acc: {val_acc:.4f})")
    
    print(f"\nEğitim tamamlandı. En iyi val accuracy: {best_val_acc:.4f}")
    
    # History kaydet
    history_path = args.out.replace(".pth", "_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History kaydedildi: {history_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model eğitimi")
    parser.add_argument("--data", default="data/synthetic", help="Veri dizini")
    parser.add_argument("--out", default="outputs/model.pth", help="Model çıktı dosyası")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch sayısı")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--n_classes", type=int, default=4, help="Sınıf sayısı")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Pretrained kullan")
    parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "resnet"], 
                       help="Backbone model")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    
    args = parser.parse_args()
    train(args)

