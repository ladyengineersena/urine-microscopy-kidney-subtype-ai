# İdrar Sedimenti Mikroskopi Görüntüleri ile Böbrek Hastalığı Alt Tipi Sınıflandırması

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Bu proje, idrar sedimenti mikroskopi görüntülerini (eritrosit silendirleri, lökosit, kristaller, epitelyal hücre tipleri vb.) kullanarak böbrek hastalığı alt tiplerini otomatik olarak sınıflandıran bir derin öğrenme modeli geliştirmektedir.

## ⚠️ ÖNEMLİ UYARI

**Bu sistem karar destek amaçlıdır; asla tek başına klinik karar vermez.**

- Araştırma ve öğretim amaçlıdır
- Klinik karar için kullanılmaz
- Gerçek hasta verileri GitHub'a yüklenmez
- Detaylı etik bilgiler için [ETHICS.md](ETHICS.md) dosyasına bakın

## Özellikler

- 🔬 **Field-level sınıflandırma**: Mikroskop alanı başına multi-class sınıflandırma
- 🧠 **Transfer Learning**: EfficientNet-B0 ve ResNet-50 ile fine-tuning
- 📊 **Model değerlendirme**: Kapsamlı metrikler ve confusion matrix
- 🔍 **Açıklanabilirlik**: Grad-CAM ile görselleştirme
- 🎨 **Sentetik veri üretimi**: Demo için sentetik görüntü üretici
- 📈 **MIL desteği**: Çoklu görüntü → hasta-level sınıflandırma (opsiyonel)

## Kurulum

### Gereksinimler

- Python 3.8+
- CUDA (GPU için, opsiyonel)

### Adımlar

1. Repository'yi klonlayın:
```bash
git clone <repository-url>
cd urine-microscopy-kidney-subtype-ai
```

2. Sanal ortam oluşturun (önerilir):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

## Hızlı Başlangıç

### 1. Sentetik Veri Üretimi

```bash
python src/data/generator_synthetic.py --out data/synthetic --n 200
```

Bu komut her sınıf için 200 sentetik görüntü üretir.

### 2. Model Eğitimi

```bash
python src/train.py \
    --data data/synthetic \
    --out outputs/model.pth \
    --epochs 10 \
    --batch 16 \
    --lr 1e-4 \
    --n_classes 4 \
    --backbone efficientnet
```

### 3. Model Değerlendirme

```bash
python src/evaluate.py \
    --data data/synthetic \
    --model outputs/model.pth \
    --n_classes 4 \
    --plot
```

### 4. Grad-CAM Görselleştirme

```bash
python src/explain.py \
    --model outputs/model.pth \
    --data data/synthetic \
    --n_samples 5 \
    --output outputs/gradcam
```

## Proje Yapısı

```
urine-microscopy-kidney-subtype-ai/
├── data/
│   ├── synthetic/              # Sentetik demo görüntüler
│   └── README_DATA.md         # Veri kullanım notları
├── src/
│   ├── data/
│   │   ├── generator_synthetic.py  # Sentetik veri üretici
│   │   └── dataset.py              # PyTorch Dataset
│   ├── models/
│   │   ├── cnn_model.py           # CNN modeli (transfer learning)
│   │   └── mil_model.py           # MIL modeli (opsiyonel)
│   ├── train.py                   # Eğitim scripti
│   ├── evaluate.py                # Değerlendirme scripti
│   └── explain.py                 # Grad-CAM açıklanabilirlik
├── notebooks/
│   └── 01_exploratory.ipynb       # Keşifsel analiz (opsiyonel)
├── outputs/                       # Model çıktıları
├── requirements.txt
├── README.md
├── ETHICS.md                      # Etik dokümantasyon
└── LICENSE
```

## Model Mimarisi

### Field-Level Sınıflandırma

- **Backbone**: EfficientNet-B0 veya ResNet-50 (ImageNet pretrained)
- **Classifier**: Dropout + Linear layer
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam with learning rate scheduling

### Ön İşleme

- Resize: 224×224
- Normalization: ImageNet mean/std
- Augmentation (training): Rotation, flip, color jitter

### Değerlendirme Metrikleri

- Accuracy
- Per-class Precision/Recall
- Macro F1-score
- Confusion Matrix
- Per-class Accuracy

## Kullanım Örnekleri

### Farklı Backbone ile Eğitim

```bash
python src/train.py --backbone resnet --data data/synthetic
```

### Özel Hyperparameter'ler

```bash
python src/train.py \
    --epochs 20 \
    --batch 32 \
    --lr 5e-5 \
    --weight_decay 1e-4
```

### Tek Görüntü için Grad-CAM

```bash
python src/explain.py \
    --model outputs/model.pth \
    --image path/to/image.png \
    --output outputs/gradcam_single.png
```

## Sınıflandırma Hedefleri

Model şu böbrek hastalığı alt tiplerini sınıflandırmayı hedefler:

- **class_0**: Minimal değişiklik hastalığı
- **class_1**: Glomerülonefrit tip A
- **class_2**: Tübüler hasar
- **class_3**: İnterstisyel nefrit

**Not**: Bu sınıflar örnek amaçlıdır. Gerçek kullanımda klinik uzmanlar ile belirlenmelidir.

## Gelecek Geliştirmeler

- [ ] Slide-level / Patient-level sınıflandırma (MIL)
- [ ] Multimodal fusion (görüntü + klinik metadata)
- [ ] Few-shot learning desteği
- [ ] Vision Transformer (ViT) implementasyonu
- [ ] Web arayüzü (Streamlit/Gradio)
- [ ] Docker containerization

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## Referanslar

- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Grad-CAM: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)
- Attention-based MIL: [Ilse et al., 2018](https://arxiv.org/abs/1802.04712)

## İletişim

Sorularınız için lütfen issue açın veya proje yöneticisi ile iletişime geçin.

## Teşekkürler

Bu proje araştırma ve öğretim amaçlı geliştirilmiştir. Tüm katkıda bulunanlara teşekkürler.

---

**Unutmayın**: Bu sistem karar destek amaçlıdır. Klinik kararlar mutlaka uzman hekimler tarafından verilmelidir.

