# src/data/generator_synthetic.py
"""
Sentetik idrar sedimenti mikroskopi görüntüleri üretici.
Gerçek hasta verileri kullanılmaz - yalnızca demo/öğretim amaçlı sentetik örnekler.
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import argparse


def make_texture(img_size=(224, 224)):
    """Temel doku oluştur (mikroskop arka planı benzeri)"""
    base = (np.random.rand(*img_size, 3) * 255).astype('uint8')
    return Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=1))


def draw_random_cell(img, kind=0):
    """
    Rastgele hücre/nesne çiz
    
    kind=0: Eritrosit benzeri (kırmızımsı yuvarlak)
    kind=1: Lökosit benzeri (açık renkli yuvarlak)
    kind=2: Silendir benzeri (uzun dikdörtgen)
    kind=3: Epitelyal hücre benzeri (dikdörtgen)
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x = np.random.randint(20, w - 20)
    y = np.random.randint(20, h - 20)
    r = np.random.randint(6, 15)
    
    if kind == 0:  # RBC-like (eritrosit)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(180, 30, 30))
    elif kind == 1:  # WBC-like (lökosit)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(220, 200, 200))
    elif kind == 2:  # cast-like (silendir - uzun)
        length = np.random.randint(15, 30)
        draw.rectangle((x - 2, y - length, x + 2, y + length), fill=(100, 80, 80))
    else:  # Epitelyal hücre
        draw.rectangle((x - r, y - r, x + r, y + r), fill=(200, 200, 100))
    
    return img


def generate(out_dir="data/synthetic", n_per_class=100):
    """
    Sentetik veri üret
    
    Args:
        out_dir: Çıktı dizini
        n_per_class: Her sınıf için görüntü sayısı
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Sınıf isimleri (böbrek hastalığı alt tipleri - örnek)
    class_names = [
        "class_0",  # Örnek: Minimal değişiklik hastalığı
        "class_1",  # Örnek: Glomerülonefrit tip A
        "class_2",  # Örnek: Tübüler hasar
        "class_3"   # Örnek: İnterstisyel nefrit
    ]
    
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(out_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        
        for i in range(n_per_class):
            img = make_texture()
            
            # Sınıfa göre farklı hücre dağılımları
            if cls_idx == 0:  # Minimal değişiklik - az hücre
                n_cells = np.random.randint(1, 5)
                cell_kinds = np.random.choice([0, 1, 2, 3], size=n_cells, p=[0.3, 0.3, 0.2, 0.2])
            elif cls_idx == 1:  # Glomerülonefrit - çok eritrosit
                n_cells = np.random.randint(5, 12)
                cell_kinds = np.random.choice([0, 1, 2, 3], size=n_cells, p=[0.6, 0.2, 0.1, 0.1])
            elif cls_idx == 2:  # Tübüler hasar - çok silendir
                n_cells = np.random.randint(4, 10)
                cell_kinds = np.random.choice([0, 1, 2, 3], size=n_cells, p=[0.2, 0.2, 0.5, 0.1])
            else:  # İnterstisyel nefrit - karışık
                n_cells = np.random.randint(3, 8)
                cell_kinds = np.random.choice([0, 1, 2, 3], size=n_cells, p=[0.25, 0.35, 0.2, 0.2])
            
            for kind in cell_kinds:
                img = draw_random_cell(img, kind=kind)
            
            fname = os.path.join(cls_dir, f"img_{i:04d}.png")
            img.save(fname)
    
    print(f"Sentetik veri oluşturuldu: {out_dir}")
    print(f"Her sınıf için {n_per_class} görüntü üretildi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentetik idrar sedimenti görüntüleri üret")
    parser.add_argument("--out", default="data/synthetic", help="Çıktı dizini")
    parser.add_argument("--n", type=int, default=200, help="Her sınıf için görüntü sayısı")
    args = parser.parse_args()
    
    generate(args.out, args.n)

