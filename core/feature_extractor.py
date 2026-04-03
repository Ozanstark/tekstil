import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, layer_names=['layer2', 'layer3']):
        super(ResNetFeatureExtractor, self).__init__()
        # Load pre-trained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.layer_names = layer_names
        
        # We don't need gradients for the feature extractor
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.layers = {}
        for name, module in self.backbone.named_children():
            if name in self.layer_names:
                module.register_forward_hook(self.get_hook(name))
                
    def get_hook(self, name):
        def hook(module, input, output):
            self.layers[name] = output
        return hook

    def forward(self, x):
        self.layers.clear()
        _ = self.backbone(x)
        return self.layers

def extract_features(model, dataloader, device):
    """
    Extracts features for all images in the dataloader.
    Returns a dictionary mapping image_paths to their features.
    """
    model.eval()
    model.to(device)
    
    features_dict = {}
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            features = model(images)
            
            for i, path in enumerate(paths):
                # Storing layer features per image
                img_features = {layer: features[layer][i].cpu() for layer in features}
                features_dict[path] = img_features
                
    return features_dict
