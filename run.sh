#!/bin/bash

# Klasör yolu (Scriptin çalıştığı dizinde olmasını garantilemek için)
cd /Users/oes/tekstil

# Çıkış anında arka plan uygulamalarını (API) kapatmak için trap kurgusu
cleanup() {
    echo "Sistem kapatılıyor. FastAPI sunucusu durduruluyor..."
    # API_PID doluysa backend'i durdur
    if [ -n "$API_PID" ]; then
        kill $API_PID 2>/dev/null
    fi
    exit
}
trap cleanup SIGINT SIGTERM

echo "=========================================================="
echo "🧶 Tekstil Defect Detection Başlatılıyor (HyperGraph AI)"
echo "=========================================================="
echo ""

echo "[Adım 1] Model Eğitimi (Training) başlatılıyor..."
echo "Lütfen akademik testlerin ve başarı metriklerinin konsolda belirlenmesini bekleyin."
echo "------------------------------------------------"
# Model eğitimini bekliyoruz
PYTHONPATH=. python3 core/train.py
# Herhangi bir hata kodu fırlatırsa sistemi durdur
if [ $? -ne 0 ]; then
    echo "Model eğitiminde hata ile karşılaşıldı. Lütfen MVTec veri setinin data/mvtec içinde olduğundan emin olun."
    exit 1
fi
echo "------------------------------------------------"
echo "[Model Eğitimi Başarılı!]"
echo ""

echo "[Adım 2] Yapay Zeka API Sunucusu (FastAPI) arka planda başlatılıyor..."
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# FastAPI'nin ayağa kalkması için 3 saniye bekle
sleep 3

echo ""
echo "[Adım 3] Görsel Arayüz (Streamlit Web UI) başlatılıyor..."
echo "Tarayıcınız otomatik olarak açılacaktır. Menüden fotoğraf yükleyip sonuçları test edebilirsiniz."
echo "------------------------------------------------"

# Streamlit ön planda başlatılıyor. Kullanıcı bu pencereyi kapattığında veya CTRL+C yaptığında trap çalışacak.
python3 -m streamlit run frontend/dashboard.py

# Eğer streamlit hatayla çökerse, arkada FastAPI asılı kalmasın diye cleanup çağırılır
cleanup
