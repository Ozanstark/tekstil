#!/bin/bash

cd /Users/oes/tekstil

cleanup() {
    echo "Kapatılıyor..."
    kill $API_PID 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

echo "[1/2] Yapay Zeka API'si arka planda başlatılıyor (Port: 8000)..."
python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 2

echo "[2/2] Streamlit Dashboard (Kullanıcı Ekranı) başlatılıyor (Port: 8501)..."
python3 -m streamlit run frontend/dashboard.py

cleanup
