# Etik Dokümantasyon ve Veri Kullanım Politikası

## Genel Prensipler

Bu proje, tıbbi görüntü analizi için derin öğrenme modeli geliştirmektedir. Etik ve yasal gerekliliklere uyum kritik öneme sahiptir.

## ⚠️ ÖNEMLİ UYARILAR

### 1. Klinik Kullanım Kısıtlaması

**Bu sistem asla tek başına klinik karar vermez.**

- Sistem yalnızca karar destek amaçlıdır
- Tüm klinik kararlar uzman hekimler tarafından verilmelidir
- Model çıktıları yalnızca referans olarak kullanılabilir
- Sistemin hatalarından kaynaklanan sorumluluk kabul edilmez

### 2. Veri Kullanım Politikası

#### GitHub Repository'de Yer Almayacak Veriler

**Kesinlikle yasak:**
- Gerçek hasta görüntüleri
- Kişisel tanımlayıcı bilgiler (PII/PHI):
  - İsim, soyisim
  - TC kimlik numarası
  - Telefon numarası
  - Adres bilgileri
  - Doğum tarihi (tam tarih)
  - Hasta dosya numarası (tanımlayıcı ise)
  - E-posta adresi
  - Sigorta numarası

#### Repository'de Yer Alabilecek Veriler

- Sentetik görüntüler (demo amaçlı)
- Anonimleştirilmiş metadata şablonları (örnek CSV)
- Kod ve dokümantasyon
- Model ağırlıkları (eğer gerçek veri ile eğitilmediyse)

## Gerçek Veri Kullanımı İçin Gereklilikler

### 1. Etik Kurul Onayı (IRB/Etik Kurul)

**Zorunludur:**
- Kurumunuzun Etik Kurulu'ndan yazılı onay
- Araştırma protokolünün onaylanması
- Veri kullanım izinlerinin belgelenmesi
- Sürekli etik kurul denetimi

**Onay belgeleri:**
- Etik kurul karar numarası
- Onay tarihi
- Geçerlilik süresi
- Veri kullanım kapsamı

### 2. Hasta Rızası (Informed Consent)

**Gereklilikler:**
- Bilgilendirilmiş hasta rızası (yazılı)
- Veri kullanım amaçlarının açıklanması
- Hasta haklarının korunması
- İsteğe bağlı geri çekilme hakkı

**Rıza formunda bulunması gerekenler:**
- Araştırma amacı
- Veri kullanım şekli
- Veri saklama süresi
- Gizlilik garantileri
- İletişim bilgileri

### 3. Veri Anonimleştirme (De-identification)

#### Kaldırılması Gerekenler

1. **Doğrudan Tanımlayıcılar:**
   - İsim, soyisim
   - TC kimlik numarası
   - Telefon numarası
   - E-posta adresi
   - Adres (tam adres)

2. **Tarih Bilgileri:**
   - Doğum tarihi → Yaş veya yaş grubu
   - İşlem tarihi → Gün-sayı formatı (örn: "gün_15")
   - Tam tarihler → Yıl veya ay-yıl

3. **Diğer Tanımlayıcılar:**
   - Hasta dosya numarası (hash'lenebilir)
   - Sigorta numarası
   - Benzersiz kodlar

#### Anonimleştirme Teknikleri

1. **Kaldırma (Removal):**
   - Tanımlayıcı sütunları tamamen kaldır

2. **Genelleştirme (Generalization):**
   - Yaş → Yaş grubu (örn: 20-30, 30-40)
   - Tarih → Yıl veya ay
   - Lokasyon → Geniş bölge

3. **Hash'leme (Hashing):**
   - Benzersiz ID'ler için hash fonksiyonu
   - Tek yönlü hash (SHA-256)

4. **Perturbation:**
   - Sayısal değerlere gürültü ekleme
   - Dikkatli kullanılmalı (veri kalitesini bozmamalı)

### 4. Veri Güvenliği

#### Depolama

- Şifreleme (at-rest encryption)
- Güvenli sunucular
- Erişim kontrolleri
- Düzenli yedekleme

#### Transfer

- Şifreli transfer (TLS/SSL)
- Güvenli protokoller
- Veri Transfer Anlaşması (DTA)

#### Erişim Kontrolü

- Minimum gerekli erişim prensibi
- Kullanıcı kimlik doğrulama
- Erişim logları
- Düzenli erişim gözden geçirmesi

### 5. Veri Saklama

- Belirlenen saklama süreleri
- Süre sonunda güvenli silme
- Yedeklerin de silinmesi
- Silme belgeleri

### 6. Veri Paylaşımı

#### Veri Transfer Anlaşması (DTA)

**İçermesi gerekenler:**
- Veri kullanım amacı
- Güvenlik önlemleri
- Veri saklama süresi
- İade/silme koşulları
- Sorumluluklar

#### Açık Veri Setleri

- Sadece tam anonimleştirilmiş veriler
- Etik kurul onayı
- Açık lisans (CC0, CC-BY)
- Kullanım şartları

## Yasal Uyumluluk

### Türkiye

- **KVKK (Kişisel Verilerin Korunması Kanunu)**
  - Açık rıza
  - Veri güvenliği
  - Veri saklama süreleri
  - Veri sahibi hakları

### Uluslararası

- **GDPR (Avrupa Birliği)**
  - Veri koruma
  - Unutulma hakkı
  - Veri taşınabilirliği

- **HIPAA (ABD)**
  - PHI koruması
  - Güvenlik kuralları
  - İhlal bildirimi

## Sorumluluk Reddi

1. **Klinik Kullanım:**
   - Sistem hatalarından kaynaklanan sorumluluk kabul edilmez
   - Klinik kararlar uzman hekimlerin sorumluluğundadır

2. **Veri Güvenliği:**
   - Veri ihlali durumunda yasal sorumluluk kullanıcıya aittir
   - Güvenlik önlemleri kullanıcı tarafından uygulanmalıdır

3. **Model Doğruluğu:**
   - Model performansı garanti edilmez
   - Farklı veri setlerinde farklı sonuçlar alınabilir

## Etik Kurul İletişimi

Gerçek veri kullanımı için:

1. Kurumunuzun Etik Kurulu ile iletişime geçin
2. Araştırma protokolünü hazırlayın
3. Gerekli onayları alın
4. Sürekli denetim sağlayın

## Kaynaklar

- [KVKK Kanunu](https://kvkk.gov.tr/)
- [HIPAA Privacy Rule](https://www.hhs.gov/hipaa/index.html)
- [GDPR](https://gdpr.eu/)
- [WMA Helsinki Deklarasyonu](https://www.wma.net/policies-post/wma-declaration-of-helsinki-ethical-principles-for-medical-research-involving-human-subjects/)

## Güncellemeler

Bu dokümantasyon düzenli olarak güncellenir. Son güncelleme: 2024

## İletişim

Etik sorularınız için lütfen proje yöneticisi ile iletişime geçin.

---

**Son Not**: Bu dokümantasyon yasal tavsiye değildir. Yasal gereklilikler için mutlaka hukuk danışmanınıza başvurun.

