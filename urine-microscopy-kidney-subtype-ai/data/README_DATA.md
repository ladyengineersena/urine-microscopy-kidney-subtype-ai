# Veri Kullanımı ve Etik Notlar

## ⚠️ ÖNEMLİ UYARI

**Bu repository'de gerçek hasta verileri bulunmaz ve bulunmayacaktır.**

## Sentetik Veri

Proje demo ve öğretim amaçlı sentetik görüntüler içerir. Bu görüntüler gerçek hasta verilerini temsil etmez ve klinik karar için kullanılamaz.

### Sentetik Veri Üretimi

Sentetik veri üretmek için:

```bash
python src/data/generator_synthetic.py --out data/synthetic --n 200
```

Bu komut her sınıf için 200 sentetik görüntü üretir.

## Gerçek Veri Kullanımı (Araştırma İçin)

Eğer gerçek hasta verileri ile çalışacaksanız, **mutlaka** aşağıdaki adımları takip edin:

### 1. Etik Kurul Onayı (IRB/Etik Kurul)

- Kurumunuzun Etik Kurulu'ndan yazılı onay alın
- Araştırma protokolünü onaylatın
- Veri kullanım izinlerini belgeleyin

### 2. Hasta Rızası

- Bilgilendirilmiş hasta rızası (informed consent) alın
- Veri kullanım amaçlarını açıklayın
- Hasta haklarını koruyun

### 3. Veri Anonimleştirme (De-identification)

**Kesinlikle kaldırılması gerekenler:**
- İsim, soyisim
- TC kimlik numarası
- Telefon numarası
- Adres bilgileri
- Doğum tarihi (tam tarih)
- Hasta dosya numarası (eğer tanımlayıcı ise)

**Anonimleştirme teknikleri:**
- Tarihleri gün-sayı formatına çevirin (örn: "2024-01-15" → "gün_15")
- Yaş bilgisi kullanılabilir (doğum tarihi yerine)
- Coğrafi bilgileri geniş bölgelere çevirin
- Benzersiz tanımlayıcılar oluşturun (hash kullanarak)

### 4. Veri Paylaşım Protokolleri

- Veri Transfer Anlaşması (DTA) imzalayın
- Veri güvenliği protokollerini uygulayın
- Şifreleme kullanın (at-rest ve in-transit)
- Erişim kontrolleri uygulayın

### 5. Veri Depolama

- Güvenli, şifrelenmiş depolama kullanın
- Erişim loglarını tutun
- Düzenli yedekleme yapın
- Veri saklama sürelerine uyun

### 6. GitHub'a Yükleme

**ASLA yüklemeyin:**
- Gerçek hasta görüntüleri
- Hasta metadata'sı (isim, TC, vb.)
- Tanımlanabilir bilgiler

**Yüklenebilir:**
- Sentetik görüntüler
- Anonimleştirilmiş metadata (örnek CSV şablonları)
- Kod ve dokümantasyon

## Veri Formatı

### Klasör Yapısı

```
data/
├── synthetic/              # Sentetik görüntüler
│   ├── class_0/
│   ├── class_1/
│   ├── class_2/
│   └── class_3/
└── README_DATA.md         # Bu dosya
```

### Görüntü Formatı

- Format: PNG veya JPG
- Boyut: Önerilen 224x224 (otomatik resize yapılır)
- Renk: RGB

### Metadata (Opsiyonel)

Eğer metadata kullanacaksanız, CSV formatında şu sütunları içerebilir:

```csv
image_path,class_label,age_group,gender_anonymized,date_offset
```

**Not:** Gerçek hasta bilgileri asla bu CSV'de olmamalıdır.

## Referanslar

- [HIPAA Privacy Rule](https://www.hhs.gov/hipaa/index.html)
- [GDPR](https://gdpr.eu/)
- [Türkiye Kişisel Verilerin Korunması Kanunu (KVKK)](https://kvkk.gov.tr/)

## İletişim

Veri kullanımı ile ilgili sorularınız için lütfen proje yöneticisi ile iletişime geçin.

