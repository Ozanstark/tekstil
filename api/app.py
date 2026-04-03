import os
import sys
import json
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
from torchvision import transforms
import io

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.feature_extractor import ResNetFeatureExtractor
from core.hypergraph_constructor import construct_incidence_matrix_knn
from core.hgnn_model import HGNNAnomalyDetector, incidence_to_edge_index

app = FastAPI(title="Textile Defect Detection API - HyperGraph", version="3.0.0")

# ── Global model state ──
MODEL_STATE = {
    "loaded": False,
    "hgnn": None,
    "extractor": None,
    "center": None,
    "threshold": 0.5,
    "device": "cpu",
    "transform": None,
    "metrics": None,
}


def load_model():
    if MODEL_STATE["loaded"]:
        return True

    model_path = os.path.join(PROJECT_ROOT, "models", "carpet_hgnn.pth")
    metrics_path = os.path.join(PROJECT_ROOT, "models", "carpet_metrics.json")

    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}. Run ./run.sh first.")
        return False

    device = MODEL_STATE["device"]
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Rebuild HGNN
    in_channels = checkpoint["in_channels"]
    hgnn = HGNNAnomalyDetector(in_channels=in_channels, hidden_channels=256, out_channels=128).to(device)
    hgnn.load_state_dict(checkpoint["hgnn_state_dict"])
    hgnn.eval()

    # Feature extractor
    extractor = ResNetFeatureExtractor(layer_names=["layer2", "layer3"]).to(device)
    extractor.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load metrics
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    MODEL_STATE.update({
        "loaded": True,
        "hgnn": hgnn,
        "extractor": extractor,
        "center": checkpoint["center"].to(device),
        "threshold": float(checkpoint["threshold"]),
        "transform": transform,
        "metrics": metrics,
    })
    print(f"Model loaded. Threshold: {MODEL_STATE['threshold']:.4f}")
    return True


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/")
def read_root():
    return {
        "message": "Textile Defect Detection HGNN API",
        "model_loaded": MODEL_STATE["loaded"],
        "threshold": float(MODEL_STATE["threshold"]),
        "metrics": MODEL_STATE["metrics"],
    }


@app.get("/metrics")
def get_metrics():
    """Model doğruluk metrikleri"""
    if not MODEL_STATE["metrics"]:
        return {"error": "No metrics available"}
    return MODEL_STATE["metrics"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_STATE["loaded"]:
        if not load_model():
            return JSONResponse(status_code=503, content={"error": "Model not trained yet."})

    try:
        device = MODEL_STATE["device"]
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # 1. Transform
        img_tensor = MODEL_STATE["transform"](image).unsqueeze(0).to(device)

        # 2. Patch-level feature extraction via ResNet
        with torch.no_grad():
            feat_maps = MODEL_STATE["extractor"](img_tensor)
            feat = feat_maps["layer2"]  # (1, 512, H, W)
            B, C, H, W = feat.shape
            patches = feat.reshape(B, C, H * W).permute(0, 2, 1).squeeze(0)  # (H*W, 512)

        # 3. Build per-image HyperGraph on patches & run HGNN
        with torch.no_grad():
            k = min(5, patches.shape[0])
            H_mat = construct_incidence_matrix_knn(patches, n_neighbors=k)
            edge_index = incidence_to_edge_index(H_mat).to(device)
            z, _, _ = MODEL_STATE["hgnn"](patches.to(device), edge_index)

        # 4. Anomaly score = max patch distance to normal center
        center = MODEL_STATE["center"]
        distances = torch.norm(z - center, dim=1)
        patch_scores = distances.cpu().numpy()
        
        anomaly_score = float(distances.max().item())
        mean_score = float(distances.mean().item())
        threshold = MODEL_STATE["threshold"]
        is_defective = bool(anomaly_score > threshold)

        return JSONResponse(content={
            "filename": file.filename,
            "anomaly_score": round(anomaly_score, 4),
            "mean_score": round(mean_score, 4),
            "threshold": round(threshold, 4),
            "is_defective": is_defective,
            "num_patches": int(patches.shape[0]),
            "verdict": "HATALI" if is_defective else "NORMAL",
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
