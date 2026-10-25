# DCGAN Eğitici Proje (PyTorch)

Bu README, projedeki tüm açıklamaları ve kritik noktaları tek yerde toplar. Amaç: hem **çalışan** hem de **öğretici** bir GAN projesini adım adım anlatmak.

---

## 1) GAN Nedir?
GAN (Generative Adversarial Network), iki ağın rekabet ettiği bir oyundur.

- **Generator (G):** Rastgele bir latent vektörden (gizli uzaydan) sahte görüntü üretir. Amaç: sahteyi “gerçek gibi” göstermek.
- **Discriminator (D):** Girdi görüntüsünün gerçek mi sahte mi olduğunu ayırt eder. Amaç: sahteyi yakalamak.

**Minimax sezgisi:**
G, D’yi kandırmak ister. D ise kandırılmamak ister. Bu karşılıklı yarışma, G’nin gerçek dağılıma benzeyen örnekler üretmesini sağlar.

**Matematik sezgisi:**
D’nin kaybı azalırken G’nin kaybı artar; G’nin kaybı azalırken D’nin işi zorlaşır. Bu gerilim GAN’i güçlendirir.

---

## 2) Proje Dizin Yapısı

```text
datasets/
  __init__.py
  cifar_like.py
models/
  __init__.py
  generator.py
  discriminator.py
losses/
  __init__.py
  gan_loss.py
trainer/
  __init__.py
  gan_trainer.py
train.py
```

**Neden böyle?**
- Modülerlik: Her parça ayrı evrimleşebilir.
- Sürdürülebilirlik: Deney ve araştırma için hız kazandırır.

**Alternatif:**
Her şeyi tek dosyada yazmak hızlıdır ama uzun vadede yönetilemez.

---

## 3) Dataset (CIFAR10-benzeri)
**Dosya:** `datasets/cifar_like.py`

- **Ne yapar?** CIFAR10’u indirir, 32x32 RGB görüntüleri [-1,1] aralığına normalize eder, küçük bir altküme ile eğitim sağlar.
- **Neden [-1,1]?** Generator `Tanh` çıktısı [-1,1] aralığındadır; veri uyumlu olmalıdır.
- **Alternatifler:** Custom Dataset, WebDataset, LMDB.

**Kritik Nokta:**
- `drop_last=True`: BatchNorm istatistikleri stabil kalsın diye batch boyu sabit tutulur.
- `pin_memory=True`: GPU’ya veri aktarımını hızlandırır.

---

## 4) Generator Backbone
**Dosya:** `models/generator.py`

- **Ne yapar?** Latent vektörü 32x32 RGB görüntüye dönüştürür.
- **Neden ConvTranspose?** Öğrenilebilir upsampling sağlar ve gradient akışını iyi tutar.
- **Latent vektör nedir?** Rastgele bir tohum; G bu tohumdan anlamlı görüntüler üretir.

**Kritik Noktalar:**
- **Shape akışı:** [B,z,1,1] -> [B,256,4,4] -> [B,128,8,8] -> [B,64,16,16] -> [B,3,32,32]
- **Numerical stability:** `Tanh` + normalize edilmiş veri.
- **Memory footprint:** Batch büyüdükçe GPU bellek kullanımı artar.

**Alternatifler:**
- UpSampling + Conv2d
- ResNet tabanlı Generator
- StyleGAN’da modulated conv

---

## 5) Discriminator Backbone
**Dosya:** `models/discriminator.py`

- **Ne yapar?** Görüntüyü gerçek/sahte olarak sınıflar.
- **Neden downsampling?** Daha az hesap maliyeti, daha güçlü özellik temsili.
- **Neden BatchNorm bazen yok?** D’nin erken katmanlarında BN, gerçek ve sahte istatistiklerini karıştırıp stabiliteyi bozabilir.

**Kritik Noktalar:**
- Sigmoid yok, çünkü **BCEWithLogitsLoss** numerik olarak daha stabil.
- Çıkış `logit`: D’nin güveni ham değerle ifade edilir.

**Alternatifler:**
- Spectral Norm (SN-GAN)
- PatchGAN
- ResNet D

---

## 5.1) Backbone Tasarımı Rehberi (Ne, Neden, Ne Zaman)

Bu bölüm, backbone tasarlarken **hangi katmanın ne işe yaradığı**, **ne zaman kullanıldığı** ve **kullanıldığında ne etki verdiği**ni detaylandırır.

**Tasarım hedefleri**
- Stabil gradient flow (ne G ne D “aşırı güçlü” olmamalı)
- Yeterli receptive field (model karar vermeden önce yeterince geniş alanı görmeli)
- Dengeli kapasite (G ve D güçleri dengede olmalı)
- Hesap ve bellek bütçesine uyum

### Generator (G) katmanları: ne, ne zaman, etkisi
- **ConvTranspose2d:** Öğrenilebilir upsampling. Çözünürlük artırırken kullanılır. Etki: detay üretimi iyileşir, ama checkerboard artefact riski artar.
- **Upsample + Conv2d:** ConvTranspose alternatifidir. Ne zaman: checkerboard artefact görüyorsan. Etki: daha temiz görüntü, biraz daha fazla hesap.
- **BatchNorm:** Aktivasyonları dengeler, gradient flow’u iyileştirir. Ne zaman: batch size yeterliyse (genelde 16+). Etki: stabil eğitim, daha hızlı yakınsama.
- **ReLU:** G’de yaygın. Etki: güçlü aktivasyon ve iyi gradient akışı.
- **LeakyReLU (G’de nadir):** Ne zaman: çok derin G ve “ölü ReLU” riski varsa. Etki: biraz daha stabil ama çeşitlilik artışı garanti değildir.
- **Residual bloklar:** Ne zaman: daha derin G (64x64 ve üstü). Etki: gradient akışı korunur, daha iyi detay üretimi.
- **Self-Attention:** Ne zaman: sahnede global yapı önemliyse (ör. büyük objeler). Etki: global tutarlılık artar, compute artar.
- **Tanh çıkış:** Üretilen görüntü [-1,1] aralığında olur. Etki: veri normalize ise stabil eğitim.

**G için ne zaman katman eklenir?**
- Çözünürlük artacaksa her 2x büyütme için bir upsampling bloğu eklenir.
- Dataset daha kompleks hale geldiyse (daha fazla çeşitlilik), kanal sayısı veya blok sayısı artırılır.
- Receptive field küçük kalıyorsa (objeler parçalı görünüyorsa) ekstra blok eklenir.

**G için ne zaman katman çıkarılır?**
- Küçük dataset / düşük çeşitlilik varsa (overfitting riski).
- Eğitim kararsızsa (mode collapse artıyorsa) kapasite düşürülür.
- Batch size çok küçükse BatchNorm kaldırılır veya GroupNorm/InstanceNorm düşünülür.

### Discriminator (D) katmanları: ne, ne zaman, etkisi
- **Conv2d + stride=2:** Downsampling için temel araç. Etki: daha düşük hesap ve daha yüksek seviye özellikler.
- **LeakyReLU:** Negatif bölgede gradient korur. Etki: D stabil kalır, vanishing azalır.
- **BatchNorm (opsiyonel):** Ne zaman: batch size yeterliyse ve D çok dengesiz değilse. Etki: daha hızlı yakınsama, ama gerçek/sahte istatistiği karışabilir.
- **Spectral Norm:** Ne zaman: D aşırı güçleniyorsa. Etki: Lipschitz kontrolü ile eğitim stabilize olur, ama kapasite kısmen kısıtlanır.
- **Dropout (nadiren):** Ne zaman: D aşırı ezici ise. Etki: D’yi zayıflatır, ama fazla kullanılırsa öğrenmeyi bozar.
- **Final Conv (logit):** Tek logit üretir, BCEWithLogitsLoss ile kullanılır. Etki: numerik stabilite.

**D için ne zaman katman eklenir?**
- Çözünürlük büyüdükçe daha fazla downsampling blok gerekir.
- G belirgin şekilde güçlenirse D’nin kapasitesi artırılır.

**D için ne zaman katman çıkarılır?**
- D çok güçlü ise G’ye gradient kalmaz (mode collapse riski).
- Eğitim aşırı dengesizleşirse D kapasitesi azaltılır.
- Küçük batch size varsa BN kaldırılır veya Spectral Norm tercih edilir.

### Genel kurallar ve etkileri
- **Kanal sayıları:** G’de çözünürlük arttıkça kanal sayısı genelde düşer. D’de çözünürlük düştükçe kanal sayısı artar. Etki: dengeli hesap ve güçlü temsil.
- **Kernel/stride seçimi:** `kernel=4, stride=2, padding=1` DCGAN için pratik ve dengeli bir seçimdir.
- **Weight init (DCGAN):** Erken eğitimi stabilize eder, “ölü” katmanları azaltır.
- **Norm türü seçimi:** G’de BatchNorm yaygın; D’de Spectral Norm veya InstanceNorm daha stabil olabilir.
- **Checkerboard artefact:** ConvTranspose ile görülebilir. Çözüm: Upsample+Conv2d kullanmak.
- **Receptive field:** D karar vermeden önce yeterince geniş alanı görmelidir; blok sayısı buna göre artırılır.
- **Aşırı kapasite:** Çok fazla katman overfitting ve training dengesizliği yaratır; daha küçük model bazen daha iyi sonuç verir.

### Örnek backbone konfigürasyonları (pratik şablonlar)

**Örnek: 32x32 DCGAN Generator (bu projedeki gibi)**
```text
z -> 4x4 (CT, 4*base) -> 8x8 (CT, 2*base) -> 16x16 (CT, base) -> 32x32 (CT, 3ch) -> Tanh
```

**Örnek: 64x64 Generator (bir blok daha ekle)**
```text
z -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> Tanh
```

**Örnek: 32x32 Discriminator**
```text
32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1 logit
```

### Örnek Backbone Tasarımı (katman katman açıklama)

**Örnek: 32x32 DCGAN Generator (katman bazında)**
```text
z(128) -> CT(4x4, 256ch) -> BN -> ReLU
       -> CT(8x8, 128ch) -> BN -> ReLU
       -> CT(16x16, 64ch) -> BN -> ReLU
       -> CT(32x32, 3ch) -> Tanh
```

**Katmanların ne işe yaradığı**
- **CT (ConvTranspose2d):** Öğrenilebilir upsampling. Çözünürlüğü büyütürken detay üretir.
- **BN (BatchNorm):** Aktivasyon dağılımını dengeler, gradient akışını stabil tutar.
- **ReLU:** Pozitif aktivasyonla güçlü gradient sağlar, G’nin öğrenmesini hızlandırır.
- **Tanh:** Çıkışı [-1,1] aralığına taşır, normalize veriyle uyumludur.

**Örnek: 32x32 DCGAN Discriminator (katman bazında)**
```text
32x32 -> Conv(16x16, 64ch) -> LeakyReLU
      -> Conv(8x8, 128ch) -> BN -> LeakyReLU
      -> Conv(4x4, 256ch) -> BN -> LeakyReLU
      -> Conv(1x1, 1ch) -> Logit
```

**Katmanların ne işe yaradığı**
- **Conv + stride=2:** Downsampling yapar, hesap maliyetini düşürür.
- **LeakyReLU:** Negatif bölgede gradient kaybını azaltır.
- **BN (opsiyonel):** D’yi stabilize eder, ama küçük batch’te kapatılabilir.
- **Final Conv (logit):** Tek skalar karar üretir, BCEWithLogitsLoss ile kullanılır.

### Layer sayısını neye göre ayarlamalısın?
- **Çözünürlük:** 2x büyütme/ küçültme başına bir blok eklemek temel kuraldır.
- **Veri karmaşıklığı:** Veri çeşitliliği ve detay arttıkça daha fazla blok gerekir.
- **Eğitim dengesi:** D aşırı güçleniyorsa katman sayısı azaltılır veya G artırılır.
- **Batch size:** Çok küçük batch’te fazla BN katmanı sorun yaratır, azaltılabilir.
- **Kaynak bütçesi:** GPU bellek sınırı varsa kanal sayıları veya blok sayısı düşürülür.

### Hangi tercih hangi etkiyi verir? (kısa örnekler)
- **BN eklemek:** Eğitim hızlanır, ama küçük batch’te dalgalanma artabilir.
- **SN eklemek:** D stabil hale gelir, ama aşırı kullanılırsa kapasite düşebilir.
- **Upsample+Conv:** Checkerboard azalır, ama biraz daha yavaş olabilir.
- **Residual blok:** Daha derin ağlara geçişi güvenli hale getirir.
- **Attention:** Global yapı daha tutarlı olur, compute artar.

### İnce Noktalar ve Sık Hatalar (olmazsa olmazlar)
- **Checkerboard artefact:** ConvTranspose bazı kombinasyonlarda dama tahtası üretir. `kernel=4, stride=2, padding=1` dengeli bir seçimdir. Sorun varsa `Upsample + Conv2d` dene.
- **Bias ve Norm ilişkisi:** BatchNorm varken `bias=False` tercih edilir. BN yoksa bias eklemek faydalı olabilir.
- **BatchNorm ve küçük batch:** Batch küçükse BN dengesizleşir. Alternatif: GroupNorm/InstanceNorm veya BN’yi kaldırmak.
- **WGAN-GP / R1 ile BN:** D’de BN, gradient penalty ile çatışabilir. SN veya norm’suz D daha stabil olur.
- **Receptive field yetersizliği:** D “görmeden” karar verirse eğitim bozulur. Çözünürlük arttıkça blok sayısı artırılmalı.
- **Kapasite dengesi:** D çok güçlü olursa G’ye gradient kalmaz; G çok güçlü olursa D ayırt edemez. Denge şarttır.
- **Son katmanda norm kullanma:** G’nin çıkış katmanında BN kullanılmaz; `Tanh` ile aralık sabitlenir.
- **Weight init:** DCGAN init (mean=0, std=0.02) erken stabiliteyi artırır. Kötü init “ölü” katmanlara yol açar.
- **Downsample yöntemi:** Stride conv öğrenilebilir ama aliasing yapabilir. `AvgPool + Conv` daha yumuşak sonuç verebilir.
- **Minibatch stddev (D):** Mode collapse riskini azaltır; büyük projelerde çok etkilidir.

### Sorun -> Çözüm Tablosu (gerçek örneklerle)

| Sorun (Belirti) | Olası Neden | Çözüm (Pratik Hamle) | Etkisi |
|---|---|---|---|
| Görüntülerde dama tahtası desenleri | ConvTranspose artefact | `Upsample + Conv2d` kullan | Artefact azalır, görüntü temizleşir |
| G tek tip görüntü üretmeye başlıyor | Mode collapse | D’yi biraz zayıflat, minibatch stddev ekle | Çeşitlilik artar |
| D loss çok düşük, G loss yükseliyor | D aşırı güçlü | D katman/kanal azalt, LR düşür | G’ye gradient akışı artar |
| Eğitim dengesiz, loss zıplıyor | BN küçük batch’te dengesiz | BN kaldır veya GroupNorm kullan | Stabilite artar |
| Görüntüler bulanık | Receptive field yetersiz | Ek upsample blok veya residual blok | Detay artar |
| Detay var ama global yapı bozuk | Global ilişki eksik | Self-Attention ekle | Küresel tutarlılık artar |
| Eğitim yavaş yakınsıyor | Zayıf init veya norm yok | DCGAN init + BN ekle | Yakınsama hızlanır |
| D’de aşırı overfit | D kapasitesi yüksek | Dropout düşük oran veya SN | Genelleme artar |

### Örnek: 128x128 Backbone (128/256/512 çıkışlı, çoklu ölçek)

Bu örnek, **128x128 giriş** alan ve **128, 256, 512 kanal seviyelerinde** çıktılar üreten bir backbone düzenini anlatır. Pratikte bu tip yapı, çoklu ölçek özellik çıkarmak veya FPN benzeri bir başlığa beslemek için kullanılır.

**Örnek çıktı seviyeleri (feature map boyutları):**
```text
Girdi: 128x128
Seviye-1: 64x64, 128ch
Seviye-2: 32x32, 256ch
Seviye-3: 16x16, 512ch
```

**Örnek blok akışı (discriminator/encoder tarzı):**
```text
128x128 -> Conv(64x64, 128ch) -> LeakyReLU
        -> Conv(32x32, 256ch) -> BN -> LeakyReLU
        -> Conv(16x16, 512ch) -> BN -> LeakyReLU
```

**Katmanların ne işe yaradığı**
- **Conv + stride=2:** Her blokta çözünürlüğü yarıya indirir, daha soyut özellikler çıkarır.
- **128/256/512 kanallar:** Çözünürlük düştükçe kanal sayısı artar; temsil gücü yükselir.
- **BN (opsiyonel):** Dengesizlik varsa kapatılabilir; küçük batch’te GroupNorm daha stabil olabilir.

**Ne zaman tercih edilir?**
- **Çoklu ölçek çıktı** gerekiyorsa (FPN, multi-head discriminator).
- **Yüksek çözünürlük** için derin bir temsil gerekirken.
- **Detay + global yapı** dengesini yakalamak isteniyorsa.

**Etkisi**
- **Avantaj:** Çoklu ölçekten gelen özelliklerle daha sağlam karar/veri üretimi.
- **Dezavantaj:** Hesap maliyeti ve bellek kullanımı artar.

---

## 6) Loss Fonksiyonu
**Dosya:** `losses/gan_loss.py`

- **Discriminator loss:** gerçek için hedef=1, sahte için hedef=0
- **Generator loss:** sahteyi “gerçek” gibi göstermeye çalışır, hedef=1

**Neden BCEWithLogits?**
- Sigmoid + BCE birlikte ve stabil.
- Numerical overflow riskini azaltır.

**Alternatifler:**
- WGAN / WGAN-GP
- Hinge loss
- LSGAN

---

## 7) Training Loop
**Dosya:** `trainer/gan_trainer.py`

- **Neden D ve G ayrı update?** İki farklı hedef var. Aynı anda update edilirse gradient sinyali bozulur.
- **Detach neden?** D update edilirken G’nin parametreleri etkilenmesin.

**Gradient flow sezgisi:**
- D update: `fake.detach()` ile G’ye gradient akmaz.
- G update: D üzerinden gradient akar ve G gelişir.

**Logging & Checkpoint:**
- TensorBoard ile loss takibi
- `samples/` klasörüne görsel kayıt
- Checkpoint ile eğitim devam ettirilebilir.

**Kritik Noktalar:**
- `set_to_none=True`: bellek kullanımını düşürür.
- `fixed_noise`: eğitim sürecinde ilerlemeyi görsel kıyaslamak için.
- Eğitim stabilitesi için doğru LR ve betalar (Adam, 0.5/0.999).

---

## 7.1) Metrik Hesaplama Scripti
**Dosya:** `metrics/compute_metrics.py`

Bu script **FID** ve **Inception Score** hesaplar. Küçük sample sayılarıyla **yaklaşık** sonuç verir.

**Neye yarar?**
- **FID:** Gerçek ve sahte görüntülerin feature uzayındaki dağılımlarını karşılaştırır. Düşük olması iyidir.
- **Inception Score:** Görüntülerin “anlaşılabilirliği” ve çeşitliliğini ölçer. Yüksek olması iyidir, ama tek başına yeterli değildir.

**Neden bu şekilde yazıldı?**
- CIFAR10 32x32 için hızlı değerlendirme sağlar.
- Generator çıkışı [-1,1] aralığından güvenli şekilde [0,255] formatına çevrilir.
- torchmetrics hazır ve güvenilir bir metrik altyapısı sunar.

**Alternatifler:**
- `torch-fidelity` (daha kapsamlı metrik paketi)
- `cleanfid` (FID için sade ve yaygın bir alternatif)
- Kendi Inception feature extractor’ı ile özel metrikler

**Kritik Noktalar:**
- FID/IS küçük veriyle oynak olur. 10k+ örnek önerilir.
- Generator çıktısı [-1,1] -> [0,255] çevrilir, metriklerde normalize karmaşası oluşmasın diye.
- Gerçek veriler de [0,1] -> [0,255] uint8 formatına çevrilir.
- torchmetrics bağımlılığı gerekir.

**Çalıştırma:**
```bash
pip install torchmetrics
python metrics/compute_metrics.py --ckpt runs/<run_name>/checkpoints/gan_epoch_020.pt --num-gen 10000 --real-subset 10000
```

**Satır satır (özet):**
- `parse_args`: checkpoint yolu, z-dim, batch size vb. parametreleri okur.
- `build_real_dataloader`: gerçek CIFAR10’u [0,1] aralığında yükler.
- `to_uint8`: hem gerçek hem sahte görüntüleri [0,255] uint8 formatına taşır.
- `load_generator`: checkpoint’ten generator ağırlıklarını yükler.
- `fid.update(..., real=True/False)`: gerçek ve sahte feature’ları toplayıp FID hesaplar.
- `is_metric.update(...)`: sahte görüntüler için Inception Score hesaplar.

---

## 8) Training Stabilitesi

### Mode Collapse
G yalnızca birkaç örnek üretmeye başlar.

**Çözüm önerileri:**
- WGAN-GP
- Minibatch discrimination
- Data augmentation

### Learning Rate Dengesi
D çok hızlı olursa G’ye iyi gradient veremez. G çok hızlı olursa D yetişemez.

### Label Smoothing
Gerçek etiketleri 1.0 yerine 0.9 yapmak D’yi aşırı güvenli olmaktan çıkarır.

---

## 9) StyleGAN’a Evrim

**Hangi parçalar değişir?**
- **Generator:** ConvTranspose -> modulated conv + style mapping network
- **Discriminator:** ResNet blokları + progressive downsampling
- **Loss:** Non-saturating loss + R1 regularization

**Research için kritik noktalar:**
- Regularization (R1, path length)
- Veri artırma (ADA)
- Metrikler (FID, IS)
- Latent space disentanglement

---

## 10) Dosyaların Konumu
- Dataset: `datasets/cifar_like.py`
- Generator: `models/generator.py`
- Discriminator: `models/discriminator.py`
- Loss: `losses/gan_loss.py`
- Trainer: `trainer/gan_trainer.py`
- Metrics: `metrics/compute_metrics.py`
- Entry: `train.py`

---

## 11) Çalıştırma

```bash
python train.py --epochs 20 --batch-size 128 --subset-size 10000
```

---

## 12) Kritik Noktaların Özeti (Hızlı Referans)
- **Gradient flow:** D güncellenirken G’ye gradient gitmez (`detach`).
- **Shape değişimleri:** Generator 1x1’den 32x32’ye büyütür, Discriminator 32x32’den 1 logit’e indirir.
- **Numerical stability:** `BCEWithLogitsLoss` + `Tanh` + normalize data.
- **Memory footprint:** Batch büyüdükçe bellek artar; `set_to_none=True` ve küçük batch ile yönetilebilir.

---

Bu README ileride hızlı referans için tasarlanmıştır. Eğer istersen, ek olarak:
- WGAN-GP varyantı
- StyleGAN adaptasyonu
- EMA (Exponential Moving Average)
- ADA (Adaptive Discriminator Augmentation)

ekleyebilirim.
