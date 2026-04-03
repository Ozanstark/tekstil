import streamlit as st
import requests
from PIL import Image
import io
import json

API_URL = "http://localhost:8000/predict"
API_ROOT = "http://localhost:8000/"
METRICS_URL = "http://localhost:8000/metrics"

st.set_page_config(page_title="Tekstil Hata Tespiti - HyperGraph AI", layout="wide")

# ═══════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════
st.title("🧶 Tekstil Yüzey Hata Tespiti")
st.markdown("**ResNet50 Feature Extraction → HyperGraph Neural Network (HGNN) → Anomali Tespiti**")

# ═══════════════════════════════════════════════
# SIDEBAR - Model Bilgisi & Metrikler
# ═══════════════════════════════════════════════
st.sidebar.header("📊 Model Durumu")

try:
    info = requests.get(API_ROOT, timeout=3).json()
    if info.get("model_loaded"):
        st.sidebar.success("✅ Model yüklendi ve hazır.")
        
        metrics = info.get("metrics")
        if metrics:
            st.sidebar.markdown("---")
            st.sidebar.header("📈 Model Doğruluğu")
            st.sidebar.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            st.sidebar.metric("F1 Skor", f"{metrics['best_f1']:.4f}")
            st.sidebar.metric("Eşik Değeri", f"{metrics['threshold']:.4f}")
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"**Kategori:** {metrics.get('category', 'N/A')}")
            st.sidebar.markdown(f"**Eğitim Görselleri:** {metrics.get('num_train_images', 'N/A')}")
            st.sidebar.markdown(f"**Test Görselleri:** {metrics.get('num_test_images', 'N/A')}")
            st.sidebar.markdown(f"**Patch/Görsel:** {metrics.get('num_patches_per_image', 'N/A')}")
            st.sidebar.markdown(f"**Epoch Sayısı:** {metrics.get('num_epochs', 'N/A')}")
            st.sidebar.markdown("---")
            st.sidebar.markdown(f"**Normal Ort. Skor:** {metrics.get('normal_avg_score', 0):.4f}")
            st.sidebar.markdown(f"**Hatalı Ort. Skor:** {metrics.get('defect_avg_score', 0):.4f}")
    else:
        st.sidebar.warning("⚠ Model henüz eğitilmedi.")
except Exception:
    st.sidebar.error("❌ API bağlantısı kurulamadı.")

# ═══════════════════════════════════════════════
# HyperGraph Açıklama Paneli
# ═══════════════════════════════════════════════
with st.expander("ℹ️ HyperGraph Modeli Nasıl Çalışıyor?", expanded=False):
    st.markdown("""
    ### 🔬 Adım Adım İşleyiş
    
    **1. Görüntü → Patch'lere Bölme**  
    Yüklenen 256×256 piksellik kumaş fotoğrafı, ResNet50 ağından geçirilir.
    Layer2 çıkışında 32×32 = **1024 adet patch** (doku parçası) elde edilir.
    Her patch 512 boyutlu bir özellik vektörü ile temsil edilir.
    
    **2. Patch'ler → HyperGraph Oluşturma**  
    Her patch bir **düğüm** (node) olur. Scikit-Learn KNN algoritması ile 
    birbirine en benzeyen 5 patch aynı **hiper-ayrıta** (hyperedge) bağlanır.
    Bu sayede kumaştaki tekrarlayan desen ilişkileri yakalanır.
    
    **3. HyperGraph Neural Network (HGNN)**  
    PyTorch Geometric HypergraphConv katmanları, her patch'in komşularından 
    bilgi toplamasını sağlar. Normal dokuda tüm patch'ler benzer embeddingler üretir.
    
    **4. Anomali Tespiti (Deep SVDD)**  
    Eğitim sırasında tüm normal patch embeddingllerinin bir "merkez noktaya" 
    yakınsaması öğretilir. Test sırasında merkezden uzak olan patch'ler = **HATA**.
    
    **5. Karar**  
    En anormal patch'in skoru, eğitimde hesaplanan eşik değerinden (threshold) 
    yüksekse → **"HATA TESPİT EDİLDİ"**, değilse → **"DOKU NORMAL"**.
    """)

# ═══════════════════════════════════════════════
# ANA ALAN - Görsel Yükleme ve Analiz
# ═══════════════════════════════════════════════
st.markdown("---")
uploaded_file = st.file_uploader("🖼️ Test için bir tekstil görüntüsü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Orijinal Görüntü")
        image = Image.open(uploaded_file)
        st.image(image, width=400)

    with col2:
        st.header("Analiz Sonucu")
        with st.spinner("Model (ResNet → HyperGraph → HGNN) çalıştırılıyor..."):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
                response = requests.post(API_URL, files=files, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    score = data.get("anomaly_score", 0)
                    mean_score = data.get("mean_score", 0)
                    threshold = data.get("threshold", 0.5)
                    is_defective = data.get("is_defective", False)
                    verdict = data.get("verdict", "?")
                    num_patches = data.get("num_patches", 0)

                    # Metrikler
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Maks. Anomali Skoru", f"{score:.4f}")
                    m2.metric("Ort. Anomali Skoru", f"{mean_score:.4f}")
                    m3.metric("Eşik Değeri", f"{threshold:.4f}")
                    
                    st.markdown(f"**Analiz edilen patch sayısı:** {num_patches}")

                    if is_defective:
                        st.error(f"""
                        🚨 **HATA TESPİT EDİLDİ** 
                        
                        Anomali skoru ({score:.4f}) eşik değerinin ({threshold:.4f}) **üzerinde**.
                        Bu kumaşta normal doku paterninden sapma tespit edildi.
                        """)
                    else:
                        st.success(f"""
                        ✅ **DOKU NORMAL**
                        
                        Anomali skoru ({score:.4f}) eşik değerinin ({threshold:.4f}) **altında**.
                        Kumaş doku paterni normal sınırlar içinde.
                        """)
                        
                elif response.status_code == 503:
                    st.warning("Model henüz eğitilmedi. `./run.sh` çalıştırın.")
                else:
                    st.error(f"API Hatası: HTTP {response.status_code}")
            except requests.ConnectionError:
                st.error("API sunucusuna bağlanılamıyor. `./start_ui.sh` çalıştırın.")
            except Exception as e:
                st.error(f"Beklenmeyen hata: {e}")
