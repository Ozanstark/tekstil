# HyperGraph Sinir Ağları ile Tekstil Yüzey Hatası Tespiti

**Derin Öğrenme Tabanlı Otomatik Kalite Kontrol Sistemi**

---

# BÖLÜM 1: GİRİŞ VE PROBLEMİN TANIMI

## 1.1 Problem Tanımı

Tekstil üretiminde kumaş yüzeyinde oluşan hatalar (delik, leke, renk bozulması, iplik kopması) üretim kalitesini doğrudan etkiler. Geleneksel yöntemde bu hatalar **insan gözüyle** tespit edilir. Ancak bu yaklaşımın ciddi sorunları vardır:

- **Yorgunluk:** Bir kalite kontrol uzmanı saatlerce aynı kumaşa baktığında dikkat kaybı yaşar
- **Tutarsızlık:** Farklı kişiler aynı hatayı farklı değerlendirebilir
- **Hız:** İnsan gözü, saniyede yüzlerce metre akan kumaşı kontrol edemez
- **Maliyet:** Sürekli personel istihdamı gerektiren, ölçeklenmesi zor bir süreçtir

Bu tez, bu problemleri çözmek için bir **yapay zeka sistemi** geliştirmektedir.

## 1.2 Mevcut Yöntemlerin Eksiklikleri

| Yöntem | Nasıl Çalışır | Eksikliği |
|---|---|---|
| İnsan gözü | Uzman personel kumaşı inceler | Yavaş, tutarsız, yorulma |
| Klasik görüntü işleme | Piksel eşikleme, kenar tespiti | Karmaşık desenlerde başarısız |
| Basit CNN modelleri | Evrişimli sinir ağı ile sınıflandırma | Sadece yakın komşu piksellere bakar, dokunun genel yapısını kaçırır |

## 1.3 Bu Tezin Katkısı

Bu çalışma, yukarıdaki yöntemlerden farklı olarak **HyperGraph Neural Network (Hiper-Çizge Sinir Ağı)** kullanır. Bu yaklaşımın temel avantajı şudur:

Klasik CNN'ler bir görselin sadece **yakın piksellerini** analiz eder — sanki büyüteçle kumaşın sadece küçük bir noktasına bakmak gibi. HyperGraph ise kumaştaki **tüm doku parçalarının birbiriyle ilişkisini** aynı anda değerlendirir — sanki **tüm kumaşı havadan kuşbakışı görmek** gibi.

Bir kumaştaki "normal desen" belirli bir periyodik tekrar içerir. HyperGraph bu tekrarı öğrenir ve tekrarın bozulduğu noktayı **hata** olarak işaretler.

## 1.4 Tez Yapısı

| Bölüm | İçerik |
|---|---|
| Bölüm 2 | Kullanılan yapay zeka kavramlarının teorik açıklaması |
| Bölüm 3 | Sistemin genel mimarisi ve veri akışı |
| Bölüm 4 | Kullanılan yazılım kütüphaneleri ve nedenleri |
| Bölüm 5 | Veri seti tanıtımı ve ön işleme |
| Bölüm 6 | Her kod dosyasının detaylı açıklaması |
| Bölüm 7 | Deneysel sonuçlar ve metrikler |
| Bölüm 8 | Sonuç ve gelecek çalışma önerileri |

---

# BÖLÜM 2: TEORİK ALTYAPI

## 2.1 Yapay Sinir Ağları (Artificial Neural Networks)

İnsan beyni milyarlarca **nöron** hücresinden oluşur. Her nöron, kendisine gelen sinyalleri alır, işler ve bir sonraki nörona iletir. Yapay sinir ağları bu mekanizmayı taklit eder.

**Basit anlatımla:** Bir yapay nöron birkaç sayıyı (girdi) alır, bunları ağırlıklarla çarpar, toplar ve bir sonuç üretir. Binlerce nöron yan yana geldiğinde bir **katman**, katmanlar üst üste geldiğinde bir **ağ** oluşur.

**Projemizde kullanımı:** Kumaş görseli sayısal piksellere dönüştürülür → Bu sayılar sinir ağına girdi olarak verilir → Ağ, pikseller arasındaki kalıpları (pattern) öğrenir.

## 2.2 Evrişimli Sinir Ağları — CNN (Convolutional Neural Networks)

CNN, özellikle görsel verilerde başarılı olan bir sinir ağı türüdür.

**Nasıl çalışır?** Küçük bir "filtre" (örneğin 3×3 piksellik bir pencere) görselin üzerinde kayarak gezdirilir. Her pozisyonda filtrenin altındaki pikseller ile filtrenin kendisi çarpılıp toplanır. Bu işleme **evrişim (convolution)** denir.

**Günlük hayattan benzetme:** Elinizle bir kumaşın üzerinde gezdirdiğiniz büyüteci düşünün. Büyüteç kumaşın her noktasında aynı şeyi arar — bir pürüz, bir düğüm veya bir renk farkı. CNN'deki filtreler de tam olarak bunu yapar.

**Projemizde kullanımı:** ResNet50 isimli CNN, kumaş fotoğrafını alır ve her bölgesindeki doku özelliklerini (texture features) sayısal vektörlere dönüştürür.

## 2.3 ResNet ve Transfer Öğrenme (Transfer Learning)

**ResNet50**, Microsoft tarafından 2015 yılında geliştirilen ve **1.4 milyon fotoğraf** üzerinde eğitilmiş bir CNN modelidir. Bu model kedileri, köpekleri, arabaları, çiçekleri — sayısız nesneyi tanıyabilir.

**Transfer Öğrenme şu anlama gelir:** Bu modelin daha önce milyonlarca fotoğraftan öğrendiği "görme yeteneğini" alıp kendi kumaş tespit projemizde kullanıyoruz. Sıfırdan bir model eğitmek yerine, zaten "görmeyi bilen" bir modeli kumaş fotoğraflarına uyarlıyoruz.

**Günlük hayattan benzetme:** Daha önce on binlerce farklı nesneyi görmüş tecrübeli bir göz doktorunu düşünün. Bu doktora "şimdi sadece kumaş hatalarına bak" diyorsunuz. Doktor sıfırdan başlamak yerine, yıllarca edindiği görsel tecrübeyi bu yeni göreve uyarlar.

**Projemizde kullanımı:** ResNet50'yi olduğu gibi (önceden eğitilmiş ağırlıklarıyla) kullanıyoruz. Modelin son sınıflandırma katmanını değil, ara katmanlarından çıkan **özellik haritalarını** (feature maps) alıyoruz.

## 2.4 Graph ve HyperGraph Nedir?

### Normal Çizge (Graph)

Bir çizge (graph), **düğümler** (nodes) ve onları birbirine bağlayan **kenarlardan** (edges) oluşur. Bir sosyal ağ düşünün: her kişi bir düğüm, iki kişi arasındaki arkadaşlık bir kenardır.

**Önemli kısıt:** Normal çizgede bir kenar **yalnızca 2 düğümü** bağlayabilir.

### HyperGraph (Hiper-Çizge)

HyperGraph'ta ise bir kenar (buna **hiper-ayrıt / hyperedge** denir) **ikiden fazla düğümü** aynı anda bağlayabilir. 

**Günlük hayattan benzetme:**
- Normal Çizge: "Ali ile Veli arkadaştır" (2 kişi, 1 bağ)
- HyperGraph: "Ali, Veli ve Ayşe aynı projede çalışır" (3 kişi, 1 bağ)

**Neden HyperGraph kullandık?** Bir kumaşta normal bir desen birden fazla bölgenin **aynı anda benzer** olmasını gerektirir. Örneğin bir halıdaki tekrar eden motif, kumaşın birçok farklı yerinde aynı şekilde görünmelidir. Bu çoklu ilişkiyi ancak HyperGraph ile yakalayabiliriz.

### Projemizdeki Kullanım

Kumaş fotoğrafı 1024 parçaya (patch) bölünür. Her parça bir düğüm olur. Birbirine en çok benzeyen 5 parça aynı hiper-ayrıta bağlanır. Böylece kumaştaki tekrar eden desenin yapısı bir HyperGraph olarak modellenmiş olur.

## 2.5 HyperGraph Neural Network (HGNN)

HGNN, HyperGraph yapısı üzerinde çalışan bir sinir ağıdır. Her düğüm (kumaş parçası), bağlı olduğu hiper-ayrıtlardaki diğer düğümlerden bilgi toplar ve kendi özellik vektörünü günceller.

**Basit anlatımla:** Her kumaş parçası, kendisiyle aynı hiper-ayrıtta olan diğer parçalara "Sen nasıl görünüyorsun?" diye sorar. Normal bir kumaşta herkes benzer cevap verir. Hatalı bir bölgede ise bir parçanın cevabı diğerlerinden **farklı** olur — bu farklılık **anomali** olarak tespit edilir.

**Teknik detay:** Projemizde PyTorch Geometric kütüphanesinin `HypergraphConv` katmanı kullanılmaktadır. İki adet HypergraphConv katmanı ardışık olarak uygulanmaktadır (512 → 256 → 128 boyut).

## 2.6 Anomali Tespiti ve Deep SVDD

Projemiz **gözetimisiz (unsupervised) anomali tespiti** yaklaşımını kullanır. Bu şu anlama gelir:

Modele eğitim sırasında **sadece normal (hatasız) kumaş fotoğrafları** gösterilir. Model asla bir "hata" görmez. Bunun yerine "normal nasıl görünür" öğrenir. Test sırasında, normalden sapan her şeyi **anomali (hata)** olarak işaretler.

**Deep SVDD (Support Vector Data Description):** Bu yöntemde ağ, tüm normal veri noktalarını çok boyutlu bir uzayda **tek bir merkez noktanın** etrafına toplamayı öğrenir. Test sırasında merkezden uzak olan noktalar anormal kabul edilir.

**Günlük hayattan benzetme:** Bir çobanın koyun sürüsünü düşünün. Çoban tüm koyunların belirli bir alanda (merkezde) durmasını ister. Bir koyun gruptan ayrılıp uzaklaşırsa, çoban onu fark eder. Burada koyunlar = kumaş parçaları, merkez = normal dokunun temsili, uzaklaşan koyun = hatalı bölge.

## 2.7 Değerlendirme Metrikleri

Bir modelin ne kadar iyi çalıştığını ölçmek için kullandığımız araçlar:

### ROC-AUC Skoru (0 ile 1 arası)

Modelin hatalı ürünü hatalı, normal ürünü normal olarak ayırt etme kabiliyetini ölçer.

| Değer | Anlamı |
|---|---|
| 1.00 | Mükemmel — hiç hata yapmıyor |
| 0.90+ | Çok iyi |
| 0.80–0.90 | İyi |
| 0.70–0.80 | Orta |
| 0.50 | Rastgele tahmin (yazı-tura atmak kadar) |
| 0.50 altı | Modelden kötü |

### F1 Skoru (0 ile 1 arası)

Hassasiyet (precision) ve duyarlılık (recall) arasındaki dengeyi ölçer.
- **Hassasiyet:** "Hatalı dediğim şeylerin kaçı gerçekten hatalıydı?"
- **Duyarlılık:** "Gerçekte hatalı olan şeylerin kaçını yakalayabildim?"

### Eşik Değeri (Threshold)

Model her görsel için bir anomali **skoru** üretir. Bu skor bir sayıdır. Eşik değeri, bu sayının üzerindeki her şeyi "hatalı", altındakileri "normal" olarak sınıflandırır.

**Günlük hayattan benzetme:** Ateş ölçerle vücut sıcaklığınızı ölçtüğünüzü düşünün. Doktorlar 37.5°C'yi "ateş başlangıcı" eşik değeri olarak kabul eder. 37.0 = normal, 38.0 = ateş. Bizim modelimiz de aynı şekilde çalışır: eşik altı = normal kumaş, eşik üstü = hatalı kumaş.

> **Bu Bölümde Ne Öğrendik?** Yapay sinir ağlarının nasıl çalıştığını, CNN'in görselleri nasıl analiz ettiğini, HyperGraph'ın neden normal çizgeden farklı olduğunu ve anomali tespitinin "normal öğren, farklı yakala" mantığıyla çalıştığını öğrendik.

---

# BÖLÜM 3: SİSTEM MİMARİSİ

## 3.1 Genel Mimari Şeması

Sistem üç ana katmandan oluşmaktadır:

```
┌──────────────────────────────────────────────────────────┐
│                    KULLANICI KATMANI                      │
│         Streamlit Web Arayüzü (dashboard.py)             │
│   Kullanıcı fotoğraf yükler, sonuçları ekranda görür     │
└──────────────────────────┬───────────────────────────────┘
                           │ HTTP İstekleri
┌──────────────────────────▼───────────────────────────────┐
│                    API KATMANI                            │
│            FastAPI Sunucusu (app.py)                      │
│   Gelen fotoğrafı alır, modele iletir, sonucu döndürür   │
└──────────────────────────┬───────────────────────────────┘
                           │ Model Çağrısı
┌──────────────────────────▼───────────────────────────────┐
│                 YAPAY ZEKA KATMANI                        │
│                                                          │
│  ┌─────────┐   ┌──────────┐   ┌──────┐   ┌──────────┐  │
│  │ ResNet50 │──▶│Patch'ler │──▶│HGNN  │──▶│Anomali   │  │
│  │(Özellik  │   │+HyperGraph│   │Model │   │Skoru     │  │
│  │Çıkarımı) │   │(KNN)     │   │      │   │Hesaplama │  │
│  └─────────┘   └──────────┘   └──────┘   └──────────┘  │
└──────────────────────────────────────────────────────────┘
```

## 3.2 Veri Akış Diyagramı (Adım Adım)

Bir kumaş fotoğrafının sisteme girişinden sonuç çıkışına kadarki yolculuğu:

**Adım 1 — Görüntü Alımı:**
Kullanıcı Dashboard (web ekranı) üzerinden bir kumaş fotoğrafı yükler. Bu fotoğraf API sunucusuna gönderilir.

**Adım 2 — Boyutlandırma:**
Fotoğraf 256×256 piksel boyutuna getirilir. Tüm fotoğrafların aynı boyutta olması, modelin tutarlı çalışması için gereklidir.

**Adım 3 — Normalizasyon:**
Piksel değerleri 0-255 aralığından 0-1 aralığına dönüştürülür ve ImageNet ortalama değerleriyle standartlaştırılır. Bu, ResNet'in doğru çalışması için gereklidir.

**Adım 4 — Özellik Çıkarımı (ResNet50):**
Normalize edilmiş görüntü ResNet50 ağından geçirilir. Layer2 katmanının çıkışı alınır. Bu çıktı (512 kanal × 32 × 32 boyut) bir özellik haritasıdır.

**Adım 5 — Patch'lere Bölme:**
32×32 = 1024 adet patch (doku parçası) elde edilir. Her patch, 512 boyutlu bir özellik vektörü ile temsil edilir.

**Adım 6 — HyperGraph Oluşturma:**
Scikit-Learn KNN (K-En Yakın Komşu) algoritması ile her patch'in 5 en benzer komşusu bulunur. Bu komşuluk ilişkileri, HyperGraph'ın hiper-ayrıtlarını oluşturur.

**Adım 7 — HGNN İşleme:**
1024 patch ve aralarındaki HyperGraph bağlantıları, iki adet HypergraphConv katmanından geçirilir. Her patch'in özellik vektörü 512 → 256 → 128 boyuta indirgenir.

**Adım 8 — Anomali Skoru Hesaplama:**
Her patch'in 128 boyutlu vektörü, eğitim sırasında öğrenilen "normal merkez" noktasına olan uzaklığıyla karşılaştırılır. En uzak patch'in skoru, o görselin anomali skoru olur.

**Adım 9 — Karar:**
Anomali skoru, eğitim sırasında belirlenen eşik değerinden yüksekse → "HATALI", değilse → "NORMAL".

## 3.3 Eğitim Süreci

Model eğitim sırasında **sadece normal (hatasız) kumaş fotoğrafları** görür:

1. 280 adet normal kumaş fotoğrafı sisteme verilir
2. Her fotoğraf 1024 patch'e bölünür (toplam 286.720 patch)
3. Her fotoğrafın patch'leri arası HyperGraph kurulur
4. HGNN, tüm patch embedding'lerini tek bir merkez noktaya yaklaştırmayı öğrenir
5. 20 epoch (döngü) boyunca bu süreç tekrarlanır
6. Eğitim bittiğinde "normal merkez" ve "eşik değeri" kaydedilir

## 3.4 Test/Çıkarım Süreci

Yeni bir görsel geldiğinde:

1. Aynı adımlar uygulanır (resize → normalize → ResNet → patch → HyperGraph → HGNN)
2. Her patch'in merkezden uzaklığı hesaplanır
3. En uzak patch'in skoru, eşik değeriyle karşılaştırılır
4. Sonuç Dashboard'da kullanıcıya gösterilir

> **Bu Bölümde Ne Öğrendik?** Sistemin üç katmandan (Kullanıcı, API, Yapay Zeka) oluştuğunu ve bir kumaş fotoğrafının 9 adımda nasıl analiz edildiğini öğrendik.

---

# BÖLÜM 4: KULLANILAN TEKNOLOJİLER VE KÜTÜPHANELER

Bu bölümde `requirements.txt` dosyasındaki her kütüphanenin ne iş yaptığı, projemizde nerede kullanıldığı ve neden seçildiği açıklanmaktadır.

## 4.1 torch (PyTorch)

**Ne yapar:** Facebook/Meta tarafından geliştirilen açık kaynak derin öğrenme framework'üdür. Sinir ağlarını tanımlamak, eğitmek ve çalıştırmak için kullanılır.

**Projede nerede:** Tüm model tanımları (`hgnn_model.py`), eğitim döngüsü (`train.py`), tahmin işlemleri (`app.py`).

**Neden seçildi:** Akademik araştırmalarda en yaygın kullanılan derin öğrenme aracıdır. PyTorch Geometric (HyperGraph kütüphanesi) PyTorch üzerinde çalışır.

**Alternatifi:** TensorFlow (Google), ancak HyperGraph desteği PyTorch ekosisteminde çok daha güçlüdür.

## 4.2 torchvision

**Ne yapar:** PyTorch'un görüntü işleme uzantısıdır. Hazır eğitilmiş modeller (ResNet, VGG vb.) ve görüntü dönüşüm araçları içerir.

**Projede nerede:** `feature_extractor.py` → ResNet50 modelini yükleme. `data_loader.py` ve `app.py` → Fotoğrafları yeniden boyutlandırma, sayısallaştırma, normalizasyon.

**Neden seçildi:** ResNet50'nin önceden eğitilmiş ağırlıklarını tek satır kodla yükleyebilmemizi sağlar.

## 4.3 torch-geometric (PyTorch Geometric)

**Ne yapar:** Çizge (Graph) ve HyperGraph tabanlı sinir ağları için kütüphanedir.

**Projede nerede:** `hgnn_model.py` → `HypergraphConv` katmanı bu kütüphaneden gelir. **Projenin kalbidir.**

**Neden seçildi:** Python ekosisteminde HyperGraph Neural Network implementasyonu sunan en olgun kütüphanedir. Akademik makalelerde referans olarak kullanılır.

**Alternatifi:** DGL (Deep Graph Library) — benzer yeteneklere sahiptir ancak PyTorch Geometric'in dokümantasyonu daha kapsamlıdır.

## 4.4 scikit-learn

**Ne yapar:** Klasik makine öğrenmesi algoritmaları ve değerlendirme araçları sunar.

**Projede nerede — 3 farklı görev:**

| Görev | Kullanılan Araç | Dosya |
|---|---|---|
| HyperGraph oluşturma | `NearestNeighbors` (KNN) | `hypergraph_constructor.py` |
| Model değerlendirme | `roc_auc_score`, `f1_score` | `train.py` |
| Alternatif kümeleme | `KMeans` | `hypergraph_constructor.py` |

**Neden seçildi:** 20+ yıllık geçmişe sahip, endüstri standardı bir kütüphanedir. KNN algoritması güvenilir ve hızlıdır.

## 4.5 scikit-image

**Ne yapar:** Bilimsel görüntü analiz kütüphanesidir.

**Projede nerede:** Görüntü analiz yardımcı işlevleri için bulundurulmaktadır.

**Neden seçildi:** Akademik görüntü işleme projeleri için optimizedir.

## 4.6 numpy

**Ne yapar:** Python'un sayısal hesaplama temelidir. Matris ve dizi (array) işlemleri yapar.

**Projede nerede:** `hypergraph_constructor.py` → Bağlantı matrisi (Incidence Matrix) oluşturma. `train.py` → Anomali skorları ve metrik hesaplamaları.

**Neden seçildi:** Python'da bilimsel hesaplama yapan her projenin temel bağımlılığıdır. PyTorch ve scikit-learn dahil tüm kütüphaneler numpy üzerine kuruludur.

## 4.7 pandas

**Ne yapar:** Tablo formatındaki verileri (CSV, Excel) okur ve işler.

**Projede nerede:** Sensör log verilerinin (UCI SECOM CSV dosyası) okunması ve ön işlenmesi.

**Neden seçildi:** CSV, Excel ve SQL gibi tablo verilerini okuyup manipüle etmenin en hızlı yoludur.

## 4.8 opencv-python (OpenCV)

**Ne yapar:** Görüntü ve video işleme için kullanılan kütüphanedir. Intel tarafından geliştirilmiştir.

**Projede nerede:** Görüntü okuma, renk dönüşümü ve ön işleme yardımcıları.

**Neden seçildi:** Endüstriyel görüntü işleme projelerinde açık ara en yaygın kullanılan kütüphanedir.

## 4.9 fastapi

**Ne yapar:** Modern, yüksek performanslı bir Python web API framework'üdür.

**Projede nerede:** `api/app.py` → Eğitilmiş modeli internet üzerinden erişilebilir bir servis haline getirir. `/predict` endpoint'i fotoğraf alır, sonuç döndürür.

**Neden seçildi:** Flask'a göre çok daha hızlıdır, otomatik API dokümantasyonu (Swagger) üretir ve asenkron (async) çalışmayı destekler.

## 4.10 uvicorn

**Ne yapar:** FastAPI uygulamalarını çalıştıran ASGI web sunucusudur.

**Projede nerede:** `run.sh` → API sunucusunu başlatmak için `python3 -m uvicorn api.app:app` komutu kullanılır.

**Neden seçildi:** FastAPI'nin resmi olarak önerdiği sunucudur.

## 4.11 python-multipart

**Ne yapar:** HTTP üzerinden dosya yükleme (multipart form data) desteği sağlar.

**Projede nerede:** `api/app.py` → Kullanıcının Dashboard'dan yüklediği fotoğrafın API tarafından alınabilmesi için gereklidir.

**Neden seçildi:** FastAPI'nin dosya yükleme endpoint'leri bu kütüphaneye bağımlıdır. Olmazsa fotoğraf yüklenemez.

## 4.12 streamlit

**Ne yapar:** Python koduyla interaktif web arayüzleri oluşturmayı sağlayan framework'tür.

**Projede nerede:** `frontend/dashboard.py` → Kullanıcının fotoğraf yükleyip sonuçları görebildiği web paneli.

**Neden seçildi:** Tek bir Python dosyasıyla (HTML/CSS/JS bilgisi gerekmeden) kullanılabilir bir arayüz oluşturulabilir. Veri bilimi projelerinde standarttır.

**Alternatifi:** Gradio — benzer amaçlıdır ancak Streamlit'in bileşen çeşitliliği daha fazladır.

## 4.13 Pillow (PIL)

**Ne yapar:** Python'un standart görüntü açma/kaydetme kütüphanesidir.

**Projede nerede:** `data_loader.py` → Disk'teki PNG/JPG dosyalarını açma. `app.py` → API'ye gelen fotoğrafı bellek içinde açma.

**Neden seçildi:** Python ekosisteminin varsayılan görüntü kütüphanesidir. PyTorch ve Streamlit bu kütüphaneye bağımlıdır.

## 4.14 matplotlib

**Ne yapar:** Bilimsel grafik ve görselleştirme kütüphanesidir.

**Projede nerede:** ROC eğrilerinin çizilmesi, anomali haritalarının görselleştirilmesi için kullanılmak üzere eklenmiştir.

**Neden seçildi:** Akademik yayınlarda grafik oluşturmanın standart aracıdır.

> **Bu Bölümde Ne Öğrendik?** Projedeki 14 kütüphanenin her birinin ne iş yaptığını, projede nerede kullanıldığını ve neden seçildiğini detaylı olarak öğrendik.

---

# BÖLÜM 5: VERİ SETİ VE ÖN İŞLEME

## 5.1 MVTec Anomaly Detection Dataset

MVTec AD, Münih Teknik Üniversitesi ve MVTec Software GmbH tarafından 2019 yılında yayınlanan, endüstriyel yüzey hata tespiti için en yaygın kullanılan referans veri setidir.

**İçerik:**
- 15 farklı kategori: 5 tekstil/doku (halı, deri, ızgara, tahta, fayans) + 10 nesne
- Toplam 5.354 yüksek çözünürlüklü görüntü
- Her kategori için ayrı ayrı "normal" ve çeşitli "hata tipleri" mevcut

## 5.2 Projemizde Kullanılan Kategori: Carpet (Halı)

| Özellik | Değer |
|---|---|
| Eğitim görselleri | 280 adet (tamamı hatasız / "good") |
| Test görselleri | 89 adet (normal + 5 farklı hata tipi) |
| Hata tipleri | Color (renk), Cut (kesik), Hole (delik), Metal contamination, Thread |
| Görsel boyutu | 1024×1024 piksel (orijinal) |

## 5.3 Ön İşleme Adımları

Her görüntüye uygulanan dönüşümler:

**1. Boyutlandırma (Resize):** 1024×1024 → 256×256 piksel. Bilgisayarın işlem yükünü azaltmak ve tüm görselleri aynı boyuta getirmek için yapılır.

**2. Tensöre Dönüştürme (ToTensor):** Görüntünün piksel değerleri 0-255 tam sayı aralığından 0.0-1.0 ondalık sayı aralığına dönüştürülür. Sinir ağları küçük sayılarla daha iyi çalışır.

**3. Normalizasyon (Normalize):** Her renk kanalı (Kırmızı, Yeşil, Mavi) ayrı ayrı standartlaştırılır.

| Kanal | Ortalama (Mean) | Standart Sapma (Std) |
|---|---|---|
| Kırmızı (R) | 0.485 | 0.229 |
| Yeşil (G) | 0.456 | 0.224 |
| Mavi (B) | 0.406 | 0.225 |

Bu değerler ImageNet veri setinin ortalamalarıdır. ResNet50 bu değerlerle eğitildiği için aynı normalizasyon değerlerinin kullanılması zorunludur.

## 5.4 Patch Çıkarımı

256×256 piksellik görüntü ResNet50'nin Layer2 katmanından geçtiğinde 32×32 boyutlu bir özellik haritası üretilir. Bu haritanın her bir hücresi orijinal görseldeki yaklaşık 8×8 piksellik bir bölgeye karşılık gelir.

- **Toplam patch sayısı:** 32 × 32 = 1024
- **Her patch'in özellik boyutu:** 512 (Layer2'nin kanal sayısı)
- **Sonuç:** Her görsel, 1024 × 512 boyutlu bir matrise dönüşür

## 5.5 UCI SECOM Sensör Verisi (Opsiyonel Modalite)

| Özellik | Değer |
|---|---|
| Kayıt sayısı | 1.567 adet üretim kaydı |
| Sensör sayısı | 590 farklı sensör okuması |
| Hatalı kayıt | 104 adet (%6.6) |
| Normal kayıt | 1.463 adet (%93.4) |
| Eksik veri (NaN) | 41.951 hücre |

Bu veri seti yarı-iletken üretim hattından alınan gerçek sensör okumalarını içermekte olup, multimodal (çoklu veri kaynaklı) sistem genişletmesi için hazırlanmaktadır.

> **Bu Bölümde Ne Öğrendik?** MVTec veri setinin içeriğini, görsellere uygulanan ön işleme adımlarının nedenlerini, ve bir görselin 1024 patch'e nasıl bölündüğünü öğrendik.

---

# BÖLÜM 6: MODEL GELİŞTİRME — KOD AÇIKLAMALARI

Bu bölümde projedeki her dosyanın amacı ve önemli kod bloklarının ne yaptığı, teknik olmayan bir dilde açıklanmaktadır.

## 6.1 data_loader.py — Veri Yükleyici

**Dosyanın amacı:** Disk üzerindeki kumaş fotoğraflarını okuyarak yapay zeka modeline beslenecek formata dönüştürür.

**Anahtar kavram — Dataset ve DataLoader:**
- `Dataset`: "Hangi fotoğraflar var ve neredeler?" sorusuna cevap verir
- `DataLoader`: "Bu fotoğrafları kaçar kaçar (batch) modele vereyim?" kararını alır

**Çalışma mantığı:**
1. Belirtilen klasöre (ör. `data/mvtec/carpet/train/good/`) gider
2. Tüm `.png` ve `.jpg` dosyalarını listeler
3. `good` klasöründekilere "normal" (etiket: 0), diğerlerine "anomali" (etiket: 1) etiketi atar
4. Her fotoğraf istendiğinde boyutlandırma, sayısallaştırma ve normalizasyon uygular

## 6.2 feature_extractor.py — Özellik Çıkarıcı

**Dosyanın amacı:** Önceden eğitilmiş ResNet50 modelini bir "özellik çıkarma makinesi" olarak kullanır.

**Anahtar kavram — Hook Mekanizması:**
ResNet50'nin 50 katmanı vardır. Biz son katmanın sonucunu değil, **ara katmanların çıktısını** istiyoruz. `Hook` mekanizması, modelin belirli bir katmanından geçen veriyi yakalamamızı sağlar.

**Çalışma mantığı:**
1. ResNet50 modeli, ImageNet ağırlıklarıyla (önceden eğitilmiş) yüklenir
2. Modelin ağırlıkları dondurulur (yeni bir şey öğrenmemesi için — `requires_grad = False`)
3. Layer2 ve Layer3 katmanlarına "hook" eklenir
4. Bir fotoğraf modelden geçirildiğinde, hook'lar ara çıktıları yakalar
5. Bu ara çıktılar (özellik haritaları) geri döndürülür

## 6.3 hypergraph_constructor.py — HyperGraph Oluşturucu

**Dosyanın amacı:** Patch özellik vektörlerinden bir HyperGraph yapısı oluşturur.

**Anahtar kavram — Incidence Matrix (Bağlantı Matrisi):**
Bir HyperGraph'ı bilgisayarda temsil etmek için bir matris kullanılır. Bu matrisin satırları düğümleri (patch'leri), sütunları hiper-ayrıtları temsil eder. Bir hücre 1 ise o düğüm o hiper-ayrıta dahildir.

**İki yöntem sunulmaktadır:**

**KNN Yöntemi (Projede kullanılan):**
1. Her patch'in 5 en benzer komşusu bulunur (scikit-learn NearestNeighbors)
2. Her patch, kendisi ve 5 komşusu ile birlikte bir hiper-ayrıt oluşturur
3. Sonuç: 1024×1024 boyutlu bir bağlantı matrisi

**KMeans Yöntemi (Alternatif):**
1. Tüm patch'ler K kümeye ayrılır
2. Her küme bir hiper-ayrıt olur
3. Aynı kümedeki patch'ler aynı hiper-ayrıtta yer alır

## 6.4 hgnn_model.py — HGNN Model Mimarisi

**Dosyanın amacı:** HyperGraph üzerinde çalışan sinir ağı modelini tanımlar.

**Modelin yapısı:**
```
Girdi (512 boyutlu patch vektörü)
    │
    ▼
HypergraphConv Katman 1 (512 → 256)
    │
    ▼
ReLU Aktivasyon (negatif değerleri sıfırla)
    │
    ▼
HypergraphConv Katman 2 (256 → 128)
    │
    ▼
ReLU Aktivasyon
    │
    ▼
Lineer Projeksiyon (128 → 128)
    │
    ▼
Çıktı (128 boyutlu embedding vektörü)
```

**HypergraphConv ne yapar?** Her düğüm (patch), bağlı olduğu hiper-ayrıtlardaki diğer düğümlerin bilgilerini toplar ve kendi özellik vektörünü günceller. Bu sayede her patch "çevresindeki doku hakkında bilgi sahibi" olur.

**ReLU ne yapar?** Hesaplanan değerlerden negatif olanları sıfıra eşitler. Basit ama etkili bir aktivasyon fonksiyonudur ve modelin doğrusal olmayan ilişkileri öğrenmesini sağlar.

**incidence_to_edge_index fonksiyonu:** Bağlantı matrisini (H) PyTorch Geometric'in anlayacağı formata (edge_index) dönüştürür.

## 6.5 train.py — Eğitim Boru Hattı

**Dosyanın amacı:** Tüm bileşenleri bir araya getirerek modelin eğitilmesini, test edilmesini ve kaydedilmesini sağlar.

**6 adımlı eğitim süreci:**

**ADIM 1 — Patch Özelliklerini Topla:**
Her eğitim görseli (280 adet) ResNet50'den geçirilir ve 1024 patch özelliği çıkarılır.

**ADIM 2 — HGNN Eğitimi (Deep SVDD):**
- Her görselin patch'leri arası HyperGraph kurulur
- Patch'ler HGNN'den geçirilir
- Kayıp fonksiyonu (loss): Tüm embedding'lerin merkeze olan uzaklıklarının ortalaması
- Optimizer bu kaybı azaltmak için ağırlıkları günceller
- Bu süreç 20 epoch (döngü) boyunca tekrarlanır

**ADIM 3 — Normal Merkezı Güncelle:**
Eğitim bittikten sonra, tüm normal patch embedding'lerinin ortalaması yeni "merkez" olarak belirlenir.

**ADIM 4 — Test Setinde Değerlendirme:**
Test görselleri (normal + hatalı) aynı süreçten geçirilir ve her birinin anomali skoru hesaplanır.

**ADIM 5 — Metrik Hesaplama:**
scikit-learn kullanılarak ROC-AUC, F1 skoru ve optimal eşik değeri hesaplanır.

**ADIM 6 — Kaydetme:**
Model ağırlıkları (`.pth`), merkez vektörü, eşik değeri ve metrikler (`.json`) diske kaydedilir.

## 6.6 app.py — API Sunucusu

**Dosyanın amacı:** Eğitilmiş modeli bir web servisi olarak sunarak, dış dünyadan fotoğraf kabul edip sonuç döndürmesini sağlar.

**Endpoint'ler (Erişim Noktaları):**

| Endpoint | Metod | Ne Yapar |
|---|---|---|
| `/` | GET | Model durumu ve metrik bilgisini döndürür |
| `/metrics` | GET | Detaylı doğruluk metriklerini döndürür |
| `/predict` | POST | Fotoğraf alır, analiz eder, sonuç döndürür |

**Çalışma mantığı:**
1. Sunucu başlatıldığında kaydedilmiş model dosyasını yükler
2. `/predict` endpoint'ine bir fotoğraf geldiğinde eğitim pipeline'ındaki aynı adımları uygular
3. Sonucu JSON formatında döndürür

## 6.7 dashboard.py — Kullanıcı Arayüzü

**Dosyanın amacı:** Yazılım bilgisi olmayan kullanıcıların sistemi kolayca kullanabilmesi için görsel bir web paneli sağlar.

**Arayüz bileşenleri:**
- **Sol kenar çubuğu:** Model doğruluk metrikleri (ROC-AUC, F1, eşik değeri)
- **Açılır panel:** "HyperGraph Nasıl Çalışıyor?" açıklaması
- **Ana alan:** Fotoğraf yükleme butonu + ikili görünüm (orijinal görsel / analiz sonucu)
- **Sonuç kartları:** Anomali skoru, ortalama skor, eşik değeri yan yana

## 6.8 run.sh — Otomasyon Scripti

**Dosyanın amacı:** Tüm sistemi tek bir komutla başlatır.

**Çalışma sırası:**
1. Model eğitimini çalıştırır ve bitmesini bekler
2. Eğitim başarılıysa API sunucusunu arka planda başlatır
3. 3 saniye bekleyip Streamlit Dashboard'u önde başlatır
4. Kullanıcı sistemi kapattığında (Ctrl+C) arka plandaki API'yi de otomatik durdurur

> **Bu Bölümde Ne Öğrendik?** Projedeki 8 dosyanın her birinin görevini, içerdiği anahtar kavramları ve çalışma mantığını teknik olmayan bir dilde öğrendik.

---

# BÖLÜM 7: DENEYSEL SONUÇLAR

## 7.1 Deney Ortamı

| Parametre | Değer |
|---|---|
| İşletim Sistemi | macOS |
| İşlemci | Apple (CPU üzerinde eğitim) |
| Python Sürümü | 3.9 |
| PyTorch Sürümü | 2.x |
| PyTorch Geometric | 2.6.1 |

## 7.2 Eğitim Parametreleri (Hiperparametreler)

| Parametre | Değer | Açıklama |
|---|---|---|
| Epoch sayısı | 20 | Model eğitim verisini 20 kez gördü |
| Learning rate | 0.0001 | Her adımda ağırlıkların ne kadar değişeceği |
| Batch size | 1 | Her seferde 1 görsel işlendi |
| Görsel boyutu | 256×256 | Tüm görseller bu boyuta getirildi |
| Patch sayısı | 1024 (32×32) | Her görseldeki parça sayısı |
| KNN k değeri | 5 | Her patch'in HyperGraph'taki komşu sayısı |
| HGNN katmanları | 2 | İki adet HypergraphConv katmanı |
| Embedding boyutu | 128 | Son çıktı vektör boyutu |
| Optimizer | Adam | Ağırlık güncelleme algoritması |

## 7.3 Sonuç Tablosu (MVTec Carpet Kategorisi)

| Metrik | Değer |
|---|---|
| ROC-AUC | 0.6970 |
| F1 Skoru | 0.8585 |
| Eşik Değeri (Threshold) | 0.0234 |
| Normal Görsellerin Ortalama Skoru | 0.0296 |
| Hatalı Görsellerin Ortalama Skoru | 0.0411 |

## 7.4 Sonuçların Yorumlanması

- **ROC-AUC (0.70):** Model rastgele tahminden (%50) anlamlı şekilde daha iyi performans göstermektedir. Ancak endüstriyel kullanım için %90+ hedeflenmektedir.
- **F1 Skoru (0.86):** Model, hatalı ve normal görselleri ayırt etmede %86 başarılıdır.
- **Normal vs Hatalı Ortalama Skor:** Hatalı görsellerin ortalama anomali skoru (0.0411), normal olanlardan (0.0296) %39 daha yüksektir — model doğru yönde ayrım yapabilmektedir.

## 7.5 İyileştirme Yönleri

| Yöntem | Beklenen Etki |
|---|---|
| Epoch sayısını artırma (50+) | Daha iyi merkez öğrenimi |
| Contrastive Learning loss fonksiyonu | Daha güçlü normal/anormal ayrımı |
| GPU ile eğitim | Daha hızlı eğitim, daha büyük batch |
| Birden fazla kategori ile eğitim | Genelleştirilmiş model |

---

# BÖLÜM 8: SONUÇ VE ÖNERİLER

## 8.1 Sonuç

Bu tez çalışmasında, tekstil yüzey hata tespiti için **HyperGraph Neural Network (HGNN)** tabanlı özgün bir sistem geliştirilmiştir. Sistem, klasik CNN yaklaşımlarından farklı olarak kumaş dokusundaki **parçalar arası çoklu ilişkileri** modelleyebilmektedir.

Geliştirilen sistemin temel katkıları:
- Kumaş fotoğrafının patch bazlı HyperGraph temsili ile doku peridoisitesinin yakalanması
- Deep SVDD yaklaşımı ile gözetimsiz anomali tespiti
- Full-stack Python mimarisi ile eğitimden web arayüzüne kadar uçtan uca çalışan sistem
- Tek komut ile çalıştırılabilen otomasyon altyapısı

## 8.2 Kısıtlamalar

- CPU üzerinde eğitim süresinin uzun olması
- Tek kategori (Carpet) üzerinde değerlendirme yapılmış olması
- ROC-AUC skorunun endüstriyel standartların altında kalması
- HyperGraph oluşturma aşamasındaki KNN hesaplama maliyeti

## 8.3 Gelecek Çalışmalar

1. **Multimodal Füzyon:** Görsel verilerin yanında sensör logları (IoT) ve kalite kontrol raporlarının HyperGraph füzyonu ile birleştirilmesi
2. **Contrastive Learning:** MM-HCAN (2025) makalesinden ilham alınarak modaliteler arası karşıtsal öğrenme uygulanması
3. **GPU ile Eğitim:** NVIDIA GPU kullanılarak eğitim süresinin kısaltılması ve daha büyük modellerin denenmesi
4. **Çoklu Kategori:** MVTec veri setindeki diğer doku kategorilerinde (leather, grid, tile, wood) test edilmesi
5. **Gerçek Zamanlı Uygulama:** Üretim hattına entegre edilebilecek düşük gecikmeli çıkarım sistemi

---

# EKLER

## EK-A: Kurulum Rehberi

**Gereksinimler:**
- Python 3.9 veya üzeri
- pip paket yöneticisi

**Kurulum adımları:**
```bash
# 1. Proje dizinine gidin
cd /Users/oes/tekstil

# 2. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 3. MVTec AD veri setini indirip data/mvtec/ klasörüne yerleştirin
# Kaynak: https://www.mvtec.com/company/research/datasets/mvtec-ad
# veya Kaggle: https://www.kaggle.com/datasets/ipythonx/mvtec-ad
```

## EK-B: Kullanım Kılavuzu

```bash
# Tam süreç: Eğitim + API + Dashboard
./run.sh

# Sadece arayüz (model zaten eğitilmişse)
./start_ui.sh
```

Dashboard açıldıktan sonra:
1. Tarayıcıda `http://localhost:8501` adresine gidin
2. "Tekstil görüntüsü yükleyin" bölümünden bir fotoğraf seçin
3. Analiz sonucunu ekranda görün

## EK-C: Proje Dosya Yapısı

```
tekstil/
├── core/                        # Yapay zeka çekirdek modülleri
│   ├── data_loader.py           # Veri yükleyici
│   ├── feature_extractor.py     # ResNet50 özellik çıkarıcı
│   ├── hypergraph_constructor.py # HyperGraph oluşturucu
│   ├── hgnn_model.py            # HGNN model mimarisi
│   └── train.py                 # Eğitim boru hattı
├── api/
│   └── app.py                   # FastAPI web sunucusu
├── frontend/
│   └── dashboard.py             # Streamlit kullanıcı arayüzü
├── data/
│   └── mvtec/                   # MVTec AD veri seti
├── dataucim/
│   └── uci-secom.csv            # Sensör log verisi
├── models/                      # Eğitilmiş model dosyaları
├── docs/
│   └── TEZ_DOKUMANTASYONU.md    # Bu belge
├── requirements.txt             # Python bağımlılıkları
├── run.sh                       # Tam otomasyon scripti
└── start_ui.sh                  # Sadece UI başlatma scripti
```

## KAYNAKÇA

1. Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
2. Feng, Y., You, H., Zhang, Z., Ji, R., & Gao, Y. (2019). *Hypergraph Neural Networks.* AAAI 2019.
3. Ruff, L., et al. (2018). *Deep One-Class Classification (Deep SVDD).* ICML 2018.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition (ResNet).* CVPR 2016.
5. Fey, M. & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop 2019.
6. MM-HCAN (2025). *Multimodal Hypergraph Contrastive Attention Network for Sensor Fusion.* (Referans makale)
