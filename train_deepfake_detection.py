"""
Enhanced Deepfake Detection System - Training Script
Optimized for 6GB GPU with 40K dataset (20K real + 20K fake)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import warnings
import time
import gc
warnings.filterwarnings('ignore')

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Model architectures
import timm

# Augmentation
import albumentations as A

# Metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.cuda.empty_cache()
    print("✓ GPU cache cleared")
else:
    print("⚠️ CUDA not available. Training will be very slow on CPU.")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset path
DATA_PATH = Path(r"C:\Users\Abdullah Nadeem\Documents\deep learmimg\real_vs_fake\real-vs-fake")

# Create directory structure
BASE_DIR = Path('./deepfake_project')
BASE_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = BASE_DIR / 'processed_data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
VIZ_DIR = BASE_DIR / 'visualizations'

for dir_path in [PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, VIZ_DIR]:
    dir_path.mkdir(exist_ok=True)

# Training configuration
EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
NUM_WORKERS = 4
PIN_MEMORY = True
SUBSET_SIZE = 20000  # 20K per class = 40K total

# Which models to train (set to True/False)
TRAIN_MODELS = {
    'EfficientNet-B0': True,
    'MobileNetV3': True,
    'DeiT-Tiny': True,
    'Custom-CNN-CBAM': True,
    'ViT-Small': True
}

print(f"\n{'='*60}")
print(f"TRAINING CONFIGURATION")
print(f"{'='*60}")
print(f"Dataset: 40K images (20K real + 20K fake)")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Models to train: {[k for k, v in TRAIN_MODELS.items() if v]}")
print(f"{'='*60}\n")

# ============================================================================
# FACE DETECTION
# ============================================================================

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image_path, target_size=(224, 224)):
    """Detect face using OpenCV Haar Cascade and crop to target size."""
    try:
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return cv2.resize(img_rgb, target_size)
        
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
        margin = int(0.1 * max(w, h))
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img_rgb.shape[1] - x, w + 2 * margin)
        h = min(img_rgb.shape[0] - y, h + 2 * margin)
        
        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, target_size)
        
        return face
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# ============================================================================
# DATASET PREPARATION
# ============================================================================

print("Loading dataset...")
real_path = DATA_PATH / 'train' / 'real'
fake_path = DATA_PATH / 'train' / 'fake'

real_images = list(real_path.glob('*.jpg')) + list(real_path.glob('*.png'))
fake_images = list(fake_path.glob('*.jpg')) + list(fake_path.glob('*.png'))

print(f"Found {len(real_images)} real images")
print(f"Found {len(fake_images)} fake images")

# Select subset
np.random.shuffle(real_images)
np.random.shuffle(fake_images)
selected_real = real_images[:SUBSET_SIZE]
selected_fake = fake_images[:SUBSET_SIZE]

# Create DataFrame
data = []
for img_path in selected_real:
    data.append({'path': str(img_path), 'label': 0, 'class': 'real'})
for img_path in selected_fake:
    data.append({'path': str(img_path), 'label': 1, 'class': 'fake'})

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total dataset: {len(df)} images")
print(f"Class distribution:\n{df['class'].value_counts()}")

# Train/Val/Test split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"\nTrain set: {len(train_df)} images")
print(f"Val set: {len(val_df)} images")
print(f"Test set: {len(test_df)} images")

# Save splits
train_df.to_csv(PROCESSED_DIR / 'train.csv', index=False)
val_df.to_csv(PROCESSED_DIR / 'val.csv', index=False)
test_df.to_csv(PROCESSED_DIR / 'test.csv', index=False)

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

train_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================================
# DATASET CLASS
# ============================================================================

class DeepfakeDataset(Dataset):
    def __init__(self, dataframe, transform=None, use_face_detection=True):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.use_face_detection = use_face_detection
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        label = self.df.loc[idx, 'label']
        
        if self.use_face_detection:
            image = detect_and_crop_face(img_path)
        else:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
        
        if image is None:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image, torch.tensor(label, dtype=torch.long)

# Create data loaders
print("\nCreating data loaders...")
train_dataset = DeepfakeDataset(train_df, transform=train_transform, use_face_detection=True)
val_dataset = DeepfakeDataset(val_df, transform=val_test_transform, use_face_detection=True)
test_dataset = DeepfakeDataset(test_df, transform=val_test_transform, use_face_detection=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True)

print(f"✓ Data loaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetB0Model, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class MobileNetV3Model(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetV3Model, self).__init__()
        self.model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class DeiTTinyModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DeiTTinyModel, self).__init__()
        self.model = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class ViTModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ViTModel, self).__init__()
        self.model = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# CBAM Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class CustomCNNWithCBAM(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNNWithCBAM, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam1 = CBAM(64)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam2 = CBAM(128)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam3 = CBAM(256)
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2))
        self.cbam4 = CBAM(512)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))
    def forward(self, x):
        x = self.cbam1(self.conv1(x))
        x = self.cbam2(self.conv2(x))
        x = self.cbam3(self.conv3(x))
        x = self.cbam4(self.conv4(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device, accumulation_steps=2):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    total_batches = len(loader)
    print_every = max(1, total_batches // 10)
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == total_batches:
            progress = (batch_idx + 1) / total_batches * 100
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            gpu_mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            print(f"  [{batch_idx+1}/{total_batches}] {progress:.1f}% | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | GPU: {gpu_mem:.0f}MB")
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    total_batches = len(loader)
    print_every = max(1, total_batches // 10)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                avg_loss = running_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                print(f"  [{batch_idx+1}/{total_batches}] {progress:.1f}% | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
    
    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels, all_probs

def train_model(model_name, model, train_loader, val_loader, epochs, lr, device):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'gpu_memory': []}
    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*50}")
        
        print("TRAINING:")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, 
                                           accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
        
        print("\nVALIDATION:")
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
            history['gpu_memory'].append(gpu_memory)
            print(f"\nGPU Memory Used: {gpu_memory:.2f} MB")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODELS_DIR / f'{model_name}_best.pth')
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"{model_name} Training Complete!")
    print(f"{'='*60}")
    print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Training Time: {training_time/60:.2f} minutes")
    print(f"{'='*60}\n")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'training_time': training_time
    }

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

results = {}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    # Train each model
    if TRAIN_MODELS['EfficientNet-B0']:
        model = EfficientNetB0Model().to(device)
        results['EfficientNet-B0'] = train_model('EfficientNet-B0', model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)
        del model
    
    if TRAIN_MODELS['MobileNetV3']:
        model = MobileNetV3Model().to(device)
        results['MobileNetV3'] = train_model('MobileNetV3', model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)
        del model
    
    if TRAIN_MODELS['DeiT-Tiny']:
        model = DeiTTinyModel().to(device)
        results['DeiT-Tiny'] = train_model('DeiT-Tiny', model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)
        del model
    
    if TRAIN_MODELS['Custom-CNN-CBAM']:
        model = CustomCNNWithCBAM().to(device)
        results['Custom-CNN-CBAM'] = train_model('Custom-CNN-CBAM', model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)
        del model
    
    if TRAIN_MODELS['ViT-Small']:
        model = ViTModel().to(device)
        results['ViT-Small'] = train_model('ViT-Small', model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)
        del model
    
    # Print final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - ALL MODELS")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name}: Best Val Acc = {result['best_val_acc']:.2f}% | Time = {result['training_time']/60:.1f} min")
    print("="*60)
    print(f"\n✓ Models saved in: {MODELS_DIR}")
    print(f"✓ Results saved in: {RESULTS_DIR}")
    print("\nTo evaluate models, run: python evaluate_models.py")
