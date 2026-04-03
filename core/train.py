import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neighbors import NearestNeighbors
from core.feature_extractor import ResNetFeatureExtractor
from core.hypergraph_constructor import construct_incidence_matrix_knn
from core.hgnn_model import HGNNAnomalyDetector, incidence_to_edge_index
from core.data_loader import get_dataloaders


def extract_patch_features(extractor, image_tensor, device):
    """
    Bir görseli ResNet'ten geçirip PATCH bazlı özellik vektörleri çıkarır.
    Görsel 256x256 ise layer2 çıkışı [B, 512, 32, 32] olur → 32x32 = 1024 patch.
    Her patch bir düğüm (node) olur.
    """
    with torch.no_grad():
        feat_maps = extractor(image_tensor.to(device))
        # layer2: [B, 512, H, W]
        feat = feat_maps['layer2']  # (1, 512, H', W')
        B, C, H, W = feat.shape
        # Her patch bir düğüm: (H*W, C) 
        patches = feat.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
    return patches, (H, W)


def compute_image_anomaly_score(patches, hgnn, center, device, n_neighbors=5):
    """
    Tek bir görselin patch'lerini HyperGraph'tan geçirip anomali skoru hesaplar.
    patches: (num_patches, feature_dim)
    Return: image_score (max patch deviation), patch_scores (heatmap için)
    """
    num_patches = patches.shape[0]
    k = min(n_neighbors, num_patches)
    
    # Patch'ler arası HyperGraph kur (KNN ile)
    H = construct_incidence_matrix_knn(patches, n_neighbors=k)
    edge_index = incidence_to_edge_index(H).to(device)
    
    # HyperGraph Neural Network'ten geçir
    z, _, _ = hgnn(patches.to(device), edge_index)
    
    # Her patch'in normal merkezden uzaklığı = anomali skor
    distances = torch.norm(z - center.to(device), dim=1)  # (num_patches,)
    
    # Görsel seviyesinde skor: en anormal patch'in skoru
    image_score = distances.max().item()
    patch_scores = distances.cpu().detach().numpy()
    
    return image_score, patch_scores


def train_and_evaluate(root_dir, category, num_epochs=20, device='cpu'):
    print(f"Loading data for category: {category}")
    train_loader, test_loader = get_dataloaders(root_dir, category, batch_size=1, img_size=256)
    
    print("Initializing models...")
    extractor = ResNetFeatureExtractor(layer_names=['layer2', 'layer3']).to(device)
    extractor.eval()
    
    # layer2 of ResNet50 → 512 channels
    in_channels = 512
    hgnn = HGNNAnomalyDetector(in_channels=in_channels, hidden_channels=256, out_channels=128).to(device)
    optimizer = optim.Adam(hgnn.parameters(), lr=1e-4)
    
    # ═══════════════════════════════════════════════════════
    # ADIM 1: Eğitim verilerinden patch özelliklerini topla
    # ═══════════════════════════════════════════════════════
    print("Extracting patch features from training images...")
    all_train_patches = []  # Her görselin tüm patch'leri
    
    for images, labels, paths in train_loader:
        patches, spatial_dims = extract_patch_features(extractor, images, device)
        all_train_patches.append(patches.squeeze(0))  # (num_patches, 512)
    
    print(f"  → {len(all_train_patches)} training images processed")
    print(f"  → Each image yields {all_train_patches[0].shape[0]} patches ({spatial_dims[0]}x{spatial_dims[1]} grid)")
    
    # ═══════════════════════════════════════════════════════
    # ADIM 2: HGNN Eğitimi (Deep SVDD Yaklaşımı)
    # Normal patch'lerin HyperGraph embeddings'lerini merkeze çek
    # ═══════════════════════════════════════════════════════
    print(f"\nStarting HGNN Training (Deep SVDD + HyperGraph) - {num_epochs} epochs...")
    
    # İlk merkezı hesapla (ortalamayla başla)
    with torch.no_grad():
        sample_patches = all_train_patches[0].to(device)
        H = construct_incidence_matrix_knn(sample_patches, n_neighbors=5)
        edge_index = incidence_to_edge_index(H).to(device)
        z_init, _, _ = hgnn(sample_patches, edge_index)
        center = z_init.mean(dim=0).detach()
    
    hgnn.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for patches in all_train_patches:
            optimizer.zero_grad()
            patches = patches.to(device)
            
            # HyperGraph kur
            k = min(5, patches.shape[0])
            H = construct_incidence_matrix_knn(patches, n_neighbors=k)
            edge_index = incidence_to_edge_index(H).to(device)
            
            # HGNN'den geçir
            z, _, _ = hgnn(patches, edge_index)
            
            # Deep SVDD loss: tüm normal patch embedding'lerini merkeze çek
            distances = torch.sum((z - center) ** 2, dim=1)
            loss = torch.mean(distances)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(all_train_patches)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")
    
    # ═══════════════════════════════════════════════════════
    # ADIM 3: Normal merkezı eğitim sonrası güncelle
    # ═══════════════════════════════════════════════════════
    print("\nComputing final normal center from training data...")
    hgnn.eval()
    all_embeddings = []
    with torch.no_grad():
        for patches in all_train_patches:
            patches = patches.to(device)
            k = min(5, patches.shape[0])
            H = construct_incidence_matrix_knn(patches, n_neighbors=k)
            edge_index = incidence_to_edge_index(H).to(device)
            z, _, _ = hgnn(patches, edge_index)
            all_embeddings.append(z)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    center = all_embeddings.mean(dim=0).detach()
    print(f"  → Center computed from {all_embeddings.shape[0]} patch embeddings")
    
    # ═══════════════════════════════════════════════════════
    # ADIM 4: Test seti üzerinde değerlendirme
    # ═══════════════════════════════════════════════════════
    print("\nEvaluating on test set...")
    image_scores = []
    y_true = []
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            patches, _ = extract_patch_features(extractor, images, device)
            patches = patches.squeeze(0)  # (num_patches, 512)
            
            score, _ = compute_image_anomaly_score(patches, hgnn, center, device)
            image_scores.append(score)
            y_true.append(labels.item())
    
    image_scores = np.array(image_scores)
    y_true = np.array(y_true)
    
    # ═══════════════════════════════════════════════════════
    # ADIM 5: scikit-learn ile metrikler
    # ═══════════════════════════════════════════════════════
    best_threshold = float(np.median(image_scores))
    best_f1 = 0.0
    auc = 0.0
    
    try:
        auc = roc_auc_score(y_true, image_scores)
        
        thresholds = np.linspace(image_scores.min(), image_scores.max(), 200)
        for thresh in thresholds:
            preds = (image_scores > thresh).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thresh)
    except ValueError as e:
        print(f"  Warning: {e}")
    
    print(f"\n{'='*50}")
    print(f"  📊 Evaluation Results for '{category}'")
    print(f"{'='*50}")
    print(f"  ROC-AUC Score     : {auc:.4f}")
    print(f"  Best F1 Score     : {best_f1:.4f}")
    print(f"  Optimal Threshold : {best_threshold:.4f}")
    print(f"  Normal Avg Score  : {image_scores[y_true == 0].mean():.4f}")
    print(f"  Defect Avg Score  : {image_scores[y_true == 1].mean():.4f}")
    print(f"{'='*50}")
    
    # ═══════════════════════════════════════════════════════
    # ADIM 6: Model, merkez, threshold kaydet
    # ═══════════════════════════════════════════════════════
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f'{category}_hgnn.pth')
    torch.save({
        'hgnn_state_dict': hgnn.state_dict(),
        'center': center.cpu().detach(),
        'threshold': float(best_threshold),
        'in_channels': in_channels,
        'category': category,
        'spatial_dims': spatial_dims,
    }, model_path)
    
    # Metrikleri JSON olarak kaydet (Dashboard için)
    metrics_path = os.path.join(save_dir, f'{category}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'category': category,
            'roc_auc': float(auc),
            'best_f1': float(best_f1),
            'threshold': float(best_threshold),
            'normal_avg_score': float(image_scores[y_true == 0].mean()),
            'defect_avg_score': float(image_scores[y_true == 1].mean()),
            'num_train_images': len(all_train_patches),
            'num_test_images': len(y_true),
            'num_patches_per_image': int(all_train_patches[0].shape[0]),
            'num_epochs': num_epochs,
        }, f, indent=2)
    
    print(f"\n  ✅ Model saved to {model_path}")
    print(f"  ✅ Metrics saved to {metrics_path}")
    
    return hgnn, center


if __name__ == '__main__':
    print("=" * 60)
    print("🧶 HyperGraph Neural Network - Textile Defect Detection")
    print("=" * 60)
    model, center = train_and_evaluate(
        '/Users/oes/tekstil/data/mvtec', 'carpet', num_epochs=20
    )
    print("\n✅ Training process entirely finished.")
