# Tekstil Yüzey Hata Tespiti (Defect Detection) Proje Analizi ve Prompt Rehberi

Bu doküman, kodlama bilgisi olmayan birinin bir yapay zeka asistanı kullanarak geliştirdiği **"Tekstil Yüzey Hata Tespiti (HyperGraph AI)"** projesinin detaylı bir analizini ve bu projenin ortaya çıkmasını sağlayan varsayımsal etkileşim (prompt) sürecini içermektedir.

---

## 1. Projenin Amacı ve Teknik Analizi

**Projenin Amacı:**
Bu projenin temel amacı, tekstil ürünlerindeki (özellikle halı/kumaş gibi dokulu yüzeylerdeki) üretim hatalarını, yırtıkları, lekeleri veya doku bozukluklarını yapay zeka kullanarak otomatik olarak tespit etmektir. Proje, "Öğreticili (Supervised)" yerine "Yarı-Öğreticili (Semi-Supervised) Anomali Tespiti" prensibiyle çalışır. Yani model sadece "sağlam" kumaşları görerek eğitilir, sağlam dokudan sapan herhangi bir şeyi "hata" (anomali) olarak işaretler.

**Kullanılan Gelişmiş Mimari (Neden Basit Değil?):**
Sıfır kodlama bilen birinin oluşturduğu bu sistem, aslında akademik düzeyde oldukça gelişmiş bir mimariye sahiptir:
1. **ResNet50 Özellik Çıkarımı (Feature Extraction):** Görüntü, önce önceden eğitilmiş (pre-trained) bir ResNet50 modeline verilir. Görüntü piksel piksel değil, "Patch" (yama/doku parçası) adı verilen anlamlı küçük parçalara (32x32 = 1024 adet) bölünür.
2. **HyperGraph Neural Network (HGNN):** Standart yapay sinir ağları yerine, bu patch'ler arasındaki ilişkiyi (birbirine benzeyen dokuları) bir Hiper-Grafik (KNN algoritması ile) üzerinde modeller.
3. **Deep SVDD (Support Vector Data Description):** Eğitim sırasında tüm normal doku parçalarının özelliklerini uzayda tek bir "merkeze" çeker. Test sırasında bu merkezden uzaklaşan (threshold'u aşan) bir doku parçası varsa, sistem bunu "Anomali/Hata" olarak algılar.
4. **FastAPI Backend:** Yapay zeka modelini dış dünyaya açan, tahminleri (predict) hızlıca yapan bir web sunucusu.
5. **Streamlit Frontend:** Kullanıcının tarayıcı üzerinden kolayca fotoğraf yükleyip saniyeler içinde hata raporunu (skorlar, metrikler, vizüel sonuçlar) görebildiği interaktif bir arayüz.
6. **Otomasyon (run.sh):** Tüm bu karmaşık eğitim, API başlatma ve arayüzü açma işlemlerini tek tuşla (bash script) halleden bir altyapı.

---

## 2. Sıfır Kodlama Bilgisiyle Bu Projeyi Üretmek İçin Gereken Promptlar

Hiç kod bilmeyen birinin bu seviyede (PyTorch modüllerine ayrılmış, API'si ve UI'ı olan) bir projeyi yapay zekaya (bana) yazdırabilmesi için, adım adım çok stratejik yönlendirmeler (promptlar) yapmış olması veya yapay zekanın "bana projeyi tarif et, mimariyi ben kurarım" teklifini kabul etmiş olması gerekir. 

İşte bu kod tabanını sıfırdan oluşturmak için verilmesi gereken aşamalı prompt serisi:

### Aşama 1: Konseptin Kararlaştırılması ve Literatür Talebi
Hiç kod bilmeyen biri ilk olarak ne istediğini tarif etmeli ve en modern yöntemi sormalıdır.

> **Kullanıcı:** *"Ben tekstil fabrikasında çalışıyorum. Kumaş ve halılardaki üretim hatalarını (leke, yırtık vs.) kameradan çekilen fotoğraflarla bulacak bir yapay zeka yapmak istiyorum. Hiç kodlama bilmiyorum, bana bu iş için şu an dünyada kullanılan en gelişmiş yapay zeka yöntemi nedir araştırıp söyler misin? Normal resim sınıflandırma değil, sadece sağlam kumaşları öğretip hatalıları bulmasını istiyorum (Anomali tespiti)."*
> 
> *(Yapay Zeka bu aşamada MVTec veri setini, ResNet feature extraction'ı ve Graph Neural Networks / Deep SVDD tekniklerini önerecektir.)*

### Aşama 2: Yapay Zeka Mimarisi ve Çekirdek Kodun Yazdırılması
Kullanıcı, yapay zekanın önerdiği karmaşık mimariyi (HGNN) kabul edip, modüler bir yapı istemelidir.

> **Kullanıcı:** *"Önerdiğin ResNet50 özellikleri ve HyperGraph Neural Network (HGNN) tabanlı Deep SVDD anomali tespiti mantığı harika duruyor. Lütfen bu projenin yapay zeka çekirdek kodlarını PyTorch kullanarak yaz. Kodları tek bir dosyaya yığma. `core` diye bir klasör oluştur; içinde veri yükleyici (`data_loader.py`), özellik çıkarıcı (`feature_extractor.py`), graf oluşturucu (`hypergraph_constructor.py`), modelin kendisi (`hgnn_model.py`) ve modeli çalıştırıp eğitecek `train.py` dosyaları ayrı ayrı olsun."*

### Aşama 3: Backend (Sunucu) Oluşturulması
Yapay zeka eğitildikten sonra, bunun bir sunucuya bağlanması gerekir.

> **Kullanıcı:** *"Şimdi bu eğittiğimiz modeli dışarıdan kullanılabilir hale getirmek istiyorum. Python'da `FastAPI` diye çok hızlı bir kütüphane varmış. Lütfen `api` adında bir klasör aç ve içine `app.py` yaz. Bu dosya, eğitilen modeli yüklesin ve dışarıdan bir resim gönderildiğinde (POST /predict) modeli çalıştırıp kumaşın resminde hata olup olmadığını (is_defective), anomali skorunu ve kabul edilebilir eşik değerini JSON olarak geri döndürsün."*

### Aşama 4: Kullanıcı Arayüzü (Frontend) Tasarımı
Sıfır kod bilen biri, siyah ekrandan (konsoldan) işlem yapamaz. Mutlaka görsel bir arayüz istemelidir.

> **Kullanıcı:** *"Ben konsol ekranı kullanmak istemiyorum, tarayıcıdan açılan görsel bir arayüze ihtiyacım var. Bunun için `Streamlit` kullanabilir misin? `frontend` klasörü içine `dashboard.py` yaz. Sayfanın solunda modelin metrikleri (doğruluk, f1 skor vs.) ve modelin nasıl çalıştığını anlatan eğitici bir metin olsun. Sağ tarafında ise bir fotoğraf yükleme alanı olsun. Fotoğraf yüklendiğinde arka planda çalışan FastAPI'ye göndersin ve dönen sonuca göre ekrana yeşil renkle 'DOKU NORMAL' veya kırmızı renkle 'HATA TESPİT EDİLDİ' yazsın."*

### Aşama 5: Sistem Entegrasyonu ve Tek Tuşla Çalıştırma (Otomasyon)
Kodlarını birleştirmeyi veya terminal kullanmayı bilmeyen birisi için her şeyi otomatize eden bir başlatıcı (script) şarttır.

> **Kullanıcı:** *"Tüm bu kodları klasörlere böldük ama ben bunları sırasıyla nasıl çalıştıracağımı, kütüphaneleri nasıl kuracağımı bilmiyorum. Benim için iki şey yap: 
> 1. Sistemin çalışması için gereken tüm kütüphanelerin listesini içeren bir `requirements.txt` dosyası oluştur.
> 2. `run.sh` adında bir dosya yaz. Ben sadece `./run.sh` yazdığımda bu dosya sırasıyla şunları kendi kendine yapsın: Önce `train.py`'yi çalıştırıp modeli eğitsin. O bitince arka planda gizlice `app.py`'yi (FastAPI) başlatsın. Son olarak da `dashboard.py`'yi (Streamlit) çalıştırıp tarayıcıda projeyi önüme getirsin."*

---

## Özet Değerlendirme

Bu projenin ortaya çıkması için en kritik nokta; kullanıcının **"Ben sadece kumaş hatalarını bulmak istiyorum"** diyerek kalmaması, aynı zamanda **"Bunu modüler tasarla", "API ve Arayüz olsun", "Tek tuşla çalışsın"** gibi *yazılım mimarisi yönlendirmeleri* (veya yapay zekanın bu konudaki yönlendirmelerine onay vermesi) yapmış olmasıdır. 

Kodlama bilinmese dahi, bir projenin **Veri -> Eğitim -> Sunucu (API) -> Kullanıcı Ekranı (UI)** döngüsünden oluştuğunun kavranmış olması ve ChatGPT / Claude gibi bir modele bu adımların sırasıyla talep edilmiş olması bu mükemmel mimariyi ortaya çıkarmıştır.
