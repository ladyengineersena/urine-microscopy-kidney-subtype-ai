# src/models/mil_model.py
"""
Multiple Instance Learning (MIL) modeli
Bir hasta için çoklu görüntü → slide-level veya hasta-level sınıflandırma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """
    Attention-based MIL (Ilse et al., 2018)
    
    Her patch için embedding → attention ağırlıkları → weighted aggregation
    """
    
    def __init__(self, n_classes=4, feature_dim=1280, hidden_dim=256):
        """
        Args:
            n_classes: Sınıf sayısı
            feature_dim: Patch embedding boyutu (EfficientNet-B0: 1280)
            hidden_dim: Attention hidden boyutu
        """
        super().__init__()
        
        # Attention mekanizması
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_patches, feature_dim) - bir hasta için çoklu patch embedding'leri
        
        Returns:
            logits: (batch_size, n_classes)
            attention_weights: (batch_size, n_patches) - interpretability için
        """
        # Attention ağırlıklarını hesapla
        attention_scores = self.attention(x)  # (batch, n_patches, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, n_patches, 1)
        
        # Weighted aggregation
        aggregated = (attention_weights * x).sum(dim=1)  # (batch, feature_dim)
        
        # Sınıflandırma
        logits = self.classifier(aggregated)
        
        return logits, attention_weights.squeeze(-1)


class SimpleMIL(nn.Module):
    """
    Basit MIL - majority voting veya average pooling
    """
    
    def __init__(self, n_classes=4, feature_dim=1280, aggregation="mean"):
        """
        Args:
            n_classes: Sınıf sayısı
            feature_dim: Patch embedding boyutu
            aggregation: "mean", "max", "attention"
        """
        super().__init__()
        
        self.aggregation = aggregation
        
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_patches, feature_dim)
        """
        if self.aggregation == "mean":
            aggregated = x.mean(dim=1)
        elif self.aggregation == "max":
            aggregated = x.max(dim=1)[0]
        elif self.aggregation == "attention":
            attention_weights = F.softmax(self.attention(x), dim=1)
            aggregated = (attention_weights * x).sum(dim=1)
        else:
            raise ValueError(f"Bilinmeyen aggregation: {self.aggregation}")
        
        logits = self.classifier(aggregated)
        return logits

