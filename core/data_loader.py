import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MVTecDataset(Dataset):
    """
    Standard PyTorch dataset structure for loading MVTec AD formats.
    Expects directory structure:
    - category_name/
      - train/
        - good/
      - test/
        - good/
        - defect_type_1/
        ...
    """
    def __init__(self, root_dir, category, is_train=True, transform=None):
        self.root_dir = root_dir
        self.category = category
        self.is_train = is_train
        self.transform = transform
        
        self.split_dir = 'train' if is_train else 'test'
        self.image_dir = os.path.join(self.root_dir, self.category, self.split_dir)
        
        self.image_paths = []
        self.labels = [] # 0 for normal, 1 for anomaly
        
        if not os.path.exists(self.image_dir):
            print(f"Warning: Directory {self.image_dir} does not exist.")
            return

        for class_name in os.listdir(self.image_dir):
            class_dir = os.path.join(self.image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            label = 0 if class_name == 'good' else 1
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

def get_dataloaders(root_dir, category, batch_size=32, img_size=256):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MVTecDataset(root_dir, category, is_train=True, transform=transform_train)
    test_dataset = MVTecDataset(root_dir, category, is_train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
