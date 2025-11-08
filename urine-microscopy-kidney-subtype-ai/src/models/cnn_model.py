# src/models/cnn_model.py
"""
CNN modeli - Transfer learning ile EfficientNet/ResNet kullanımı
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class SimpleFineTune(nn.Module):
    """
    Transfer learning ile fine-tune edilmiş model
    
    EfficientNet-B0 backbone kullanır (ImageNet pretrained)
    """
    
    def __init__(self, n_classes=4, pretrained=True, backbone="efficientnet"):
        """
        Args:
            n_classes: Sınıf sayısı
            pretrained: ImageNet pretrained ağırlıkları kullan
            backbone: "efficientnet" veya "resnet"
        """
        super().__init__()
        
        self.n_classes = n_classes
        self.backbone_name = backbone
        
        if backbone == "efficientnet":
            # PyTorch >= 0.13 için weights parametresi, eski versiyonlar için pretrained
            try:
                if pretrained:
                    self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = models.efficientnet_b0(weights=None)
            except AttributeError:
                # Eski PyTorch versiyonları için
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            # Classifier'ı değiştir
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, n_classes)
            )
        elif backbone == "resnet":
            # PyTorch >= 0.13 için weights parametresi, eski versiyonlar için pretrained
            try:
                if pretrained:
                    self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = models.resnet50(weights=None)
            except AttributeError:
                # Eski PyTorch versiyonları için
                self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, n_classes)
            )
        else:
            raise ValueError(f"Bilinmeyen backbone: {backbone}")
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_features(self, x):
        """Feature extraction (Grad-CAM için)"""
        if self.backbone_name == "efficientnet":
            # EfficientNet feature extraction
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        else:  # ResNet
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            return x


class VisionTransformerModel(nn.Module):
    """
    Vision Transformer modeli (opsiyonel - küçük veri setleri için dikkatli kullanılmalı)
    """
    
    def __init__(self, n_classes=4, pretrained=True):
        super().__init__()
        # timm kütüphanesi gerekli
        try:
            import timm
            self.model = timm.create_model('vit_base_patch16_224', 
                                          pretrained=pretrained, 
                                          num_classes=n_classes)
        except ImportError:
            raise ImportError("timm kütüphanesi gerekli: pip install timm")
    
    def forward(self, x):
        return self.model(x)

