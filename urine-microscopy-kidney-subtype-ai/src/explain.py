# src/explain.py
"""
Model açıklanabilirliği - Grad-CAM ile görselleştirme
"""

import torch
import torch.nn.functional as F
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
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os


class GradCAM:
    """
    Grad-CAM implementasyonu
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook kaydet
        self.hook_handle_forward = self.target_layer.register_forward_hook(self.save_activation)
        self.hook_handle_backward = self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output bir tuple, ilk elemanı al
        if grad_output and len(grad_output) > 0:
            self.gradients = grad_output[0]
        else:
            self.gradients = grad_input[0] if grad_input and len(grad_input) > 0 else None
    
    def generate_cam(self, input_image, class_idx=None):
        """
        Grad-CAM heatmap üret
        
        Args:
            input_image: (1, C, H, W) tensor
            class_idx: Hangi sınıf için CAM (None ise en yüksek tahmin)
        
        Returns:
            cam: (H, W) numpy array
        """
        self.model.eval()
        
        # Forward
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Gradients ve activations kontrolü
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients veya activations kaydedilemedi. Model doğru şekilde forward/backward çalıştırıldı mı?")
        
        # Batch dimension'ı handle et
        if len(self.gradients.shape) == 4:  # (B, C, H, W)
            gradients = self.gradients[0]  # (C, H, W)
        else:
            gradients = self.gradients  # (C, H, W)
            
        if len(self.activations.shape) == 4:  # (B, C, H, W)
            activations = self.activations[0]  # (C, H, W)
        else:
            activations = self.activations  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted sum of activations
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        cam = F.relu(cam)  # ReLU
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx


def visualize_gradcam(model, image_path, class_names, device, backbone="efficientnet"):
    """Grad-CAM görselleştirme"""
    # Model için target layer bul
    if backbone == "efficientnet":
        # EfficientNet'te features bir Sequential, son bloğu al
        target_layer = model.backbone.features[-1]  # Son feature layer
    else:  # ResNet
        target_layer = model.backbone.layer4[-1]
    
    # GradCAM
    gradcam = GradCAM(model, target_layer)
    
    try:
        # Görüntü yükle
        img = Image.open(image_path).convert("RGB")
        original_img = np.array(img)
        
        # Transform
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # CAM üret
        cam, pred_class = gradcam.generate_cam(input_tensor)
        
        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        cam_resized = np.uint8(255 * cam_resized)
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        
        # Overlay
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title(f"Grad-CAM Heatmap\nPredicted: {class_names[pred_class]}")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    finally:
        # Hook'ları temizle
        if hasattr(gradcam, 'hook_handle_forward'):
            gradcam.hook_handle_forward.remove()
        if hasattr(gradcam, 'hook_handle_backward'):
            gradcam.hook_handle_backward.remove()


def explain(args):
    """Açıklanabilirlik analizi"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Dataset (sınıf isimleri için)
    dataset = UrineFieldDataset(args.data, train=False)
    class_names = dataset.get_class_names()
    
    # Model
    model = SimpleFineTune(
        n_classes=args.n_classes,
        pretrained=False,
        backbone=args.backbone
    ).to(device)
    
    # Model yükle
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Tek görüntü veya dataset'ten örnekler
    if args.image:
        # Tek görüntü
        fig = visualize_gradcam(model, args.image, class_names, device, args.backbone)
        output_path = args.output or "outputs/gradcam_example.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM kaydedildi: {output_path}")
    else:
        # Dataset'ten rastgele örnekler
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        os.makedirs(args.output or "outputs/gradcam", exist_ok=True)
        
        for i, (img_tensor, label) in enumerate(loader):
            if i >= args.n_samples:
                break
            
            # Görüntüyü kaydet
            img_np = img_tensor[0].permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np, 0, 1)
            
            # Geçici dosya kaydet
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                Image.fromarray((img_np * 255).astype('uint8')).save(tmp_path)
            
            # Grad-CAM
            fig = visualize_gradcam(model, tmp_path, class_names, device, args.backbone)
            output_path = os.path.join(
                args.output or "outputs/gradcam",
                f"gradcam_sample_{i}_true_{class_names[label.item()]}.png"
            )
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            os.unlink(tmp_path)
            print(f"Kaydedildi: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model açıklanabilirliği - Grad-CAM")
    parser.add_argument("--model", default="outputs/model.pth", help="Model dosyası")
    parser.add_argument("--data", default="data/synthetic", help="Veri dizini (örnekler için)")
    parser.add_argument("--image", help="Tek görüntü yolu (opsiyonel)")
    parser.add_argument("--output", help="Çıktı dizini/dosyası")
    parser.add_argument("--n_samples", type=int, default=5, help="Örnek sayısı")
    parser.add_argument("--n_classes", type=int, default=4, help="Sınıf sayısı")
    parser.add_argument("--backbone", default="efficientnet", choices=["efficientnet", "resnet"],
                       help="Backbone model")
    
    args = parser.parse_args()
    explain(args)

