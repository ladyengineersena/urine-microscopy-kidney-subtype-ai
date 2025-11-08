# src/evaluate.py
"""
Model değerlendirme scripti
"""

import torch
import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from src.data.dataset import UrineFieldDataset
from src.models.cnn_model import SimpleFineTune
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(args):
    """Model değerlendirme"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Dataset
    dataset = UrineFieldDataset(args.data, train=False)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    print(f"Test samples: {len(dataset)}")
    print(f"Classes: {dataset.get_class_names()}")
    
    # Model
    model = SimpleFineTune(
        n_classes=args.n_classes,
        pretrained=False,  # Eğitilmiş model yüklenecek
        backbone=args.backbone
    ).to(device)
    
    # Model ağırlıklarını yükle
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Tahminler
    all_ys = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds.tolist())
            all_ys.extend(y.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    
    # Metrikler
    accuracy = accuracy_score(all_ys, all_preds)
    print(f"\n{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{'='*60}\n")
    
    # Classification report
    class_names = dataset.get_class_names()
    print("Classification Report:")
    print(classification_report(all_ys, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_ys, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Confusion matrix görselleştirme
    if args.plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = args.model.replace(".pth", "_confusion_matrix.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nConfusion matrix kaydedildi: {plot_path}")
    
    # Per-class accuracy
    print("\nPer-class Accuracy:")
    for i, cls_name in enumerate(class_names):
        class_mask = np.array(all_ys) == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(
                np.array(all_ys)[class_mask], 
                np.array(all_preds)[class_mask]
            )
            print(f"  {cls_name}: {class_acc:.4f} ({class_mask.sum()} samples)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model değerlendirme")
    parser.add_argument("--data", default="data/synthetic", help="Test veri dizini")
    parser.add_argument("--model", default="outputs/model.pth", help="Model dosyası")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--n_classes", type=int, default=4, help="Sınıf sayısı")
    parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "resnet"],
                       help="Backbone model")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers")
    parser.add_argument("--plot", action="store_true", help="Confusion matrix görselleştir")
    
    args = parser.parse_args()
    evaluate(args)

