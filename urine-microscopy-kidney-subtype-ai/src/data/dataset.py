# src/data/dataset.py
"""
PyTorch Dataset sınıfı - idrar sedimenti görüntüleri için
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class UrineFieldDataset(Dataset):
    """
    Field-level (patch/crop) idrar sedimenti görüntü veri seti
    
    Her görüntü bir mikroskop alanını temsil eder.
    """
    
    def __init__(self, root_dir, transform=None, train=True):
        """
        Args:
            root_dir: Veri dizini (her sınıf için alt klasör içerir)
            transform: Görüntü dönüşümleri
            train: Eğitim modu (augmentation için)
        """
        self.root = root_dir
        self.items = []
        
        # Sınıf dizinlerini bul
        classes = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("class_")
        ])
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Tüm görüntüleri topla
        for cls in classes:
            cls_dir = os.path.join(root_dir, cls)
            cls_idx = self.class_to_idx[cls]
            
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.items.append((os.path.join(cls_dir, f), cls_idx))
        
        # Transform belirleme
        if transform is None:
            if train:
                # Eğitim için augmentation
                self.transform = T.Compose([
                    T.Resize((256, 256)),
                    T.RandomRotation(15),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
                ])
            else:
                # Test için sadece normalize
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path, label = self.items[idx]
        
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"Hata: {path} yüklenemedi: {e}")
            # Hata durumunda siyah görüntü döndür
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            img = self.transform(img)
        
        return img, label
    
    def get_class_names(self):
        """Sınıf isimlerini döndür"""
        return list(self.class_to_idx.keys())

