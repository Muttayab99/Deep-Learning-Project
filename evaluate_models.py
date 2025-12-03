"""
Deepfake Detection - Model Evaluation Script
Evaluates all trained models on the test set
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import timm
import albumentations as A

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path('./deepfake_project')
PROCESSED_DIR = BASE_DIR / 'processed_data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
VIZ_DIR = BASE_DIR / 'visualizations'

BATCH_SIZE = 16
NUM_WORKERS = 4

# ============================================================================
# DATASET AND TRANSFORMS
# ============================================================================

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image_path, target_size=(224, 224)):
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
        return cv2.resize(face, target_size)
    except:
        return None

test_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
# EVALUATION FUNCTIONS
# ============================================================================

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
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
    
    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels, all_probs

def compute_metrics(y_true, y_pred, y_probs):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc_roc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm
    }

# ============================================================================
# MAIN EVALUATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60 + "\n")
    
    # Load test dataset
    print("Loading test dataset...")
    test_df = pd.read_csv(PROCESSED_DIR / 'test.csv')
    test_dataset = DeepfakeDataset(test_df, transform=test_transform, use_face_detection=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test set: {len(test_dataset)} images\n")
    
    # Model configurations
    model_configs = {
        'EfficientNet-B0': EfficientNetB0Model,
        'MobileNetV3': MobileNetV3Model,
        'DeiT-Tiny': DeiTTinyModel,
        'Custom-CNN-CBAM': CustomCNNWithCBAM,
        'ViT-Small': ViTModel
    }
    
    test_results = {}
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate each trained model
    for model_name, model_class in model_configs.items():
        model_path = MODELS_DIR / f'{model_name}_best.pth'
        
        if not model_path.exists():
            print(f"⚠️  Skipping {model_name} - model not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model = model_class().to(device)
        model.load_state_dict(torch.load(model_path))
        
        # Evaluate
        test_loss, test_acc, preds, labels, probs = validate(model, test_loader, criterion, device)
        metrics = compute_metrics(labels, preds, probs)
        
        # Measure inference time
        model.eval()
        sample_batch, _ = next(iter(test_loader))
        sample_batch = sample_batch.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_batch)
        
        # Measure
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(sample_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = (time.time() - start) / 100 * 1000  # ms per batch
        inference_time_per_image = inference_time / BATCH_SIZE
        
        # Model size
        model_size = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)  # MB
        
        test_results[model_name] = {
            'test_acc': test_acc,
            'metrics': metrics,
            'inference_time_ms': inference_time_per_image,
            'model_size_mb': model_size,
            'predictions': preds,
            'labels': labels,
            'probabilities': probs
        }
        
        # Print metrics
        print(f"\nTest Accuracy: {test_acc:.2f}%")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"Inference Time: {inference_time_per_image:.2f} ms/image")
        print(f"Model Size: {model_size:.2f} MB")
        print(f"FPS: {1000/inference_time_per_image:.1f}")
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create comparison table
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE")
    print("="*100)
    
    comparison_data = []
    for model_name, result in test_results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']*100:.2f}%",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'AUC-ROC': f"{metrics['auc_roc']:.4f}",
            'Inference (ms)': f"{result['inference_time_ms']:.2f}",
            'Size (MB)': f"{result['model_size_mb']:.2f}",
            'FPS': f"{1000/result['inference_time_ms']:.1f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print("="*100)
    
    # Save comparison table
    comparison_df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    print(f"\n✓ Comparison table saved to: {RESULTS_DIR / 'model_comparison.csv'}")
    
    # Plot confusion matrices
    if len(test_results) > 0:
        print("\nGenerating confusion matrices...")
        n_models = len(test_results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, result) in enumerate(test_results.items()):
            cm = result['metrics']['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            axes[idx].set_title(f"{model_name}\nAcc: {result['test_acc']:.2f}%", fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide extra subplots
        for idx in range(len(test_results), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrices saved to: {VIZ_DIR / 'confusion_matrices.png'}")
    
    # Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Models evaluated: {len(test_results)}")
    if len(test_results) > 0:
        best_model = max(test_results.items(), key=lambda x: x[1]['test_acc'])
        print(f"Best model: {best_model[0]} ({best_model[1]['test_acc']:.2f}%)")
    print(f"\nResults saved in: {RESULTS_DIR}")
    print(f"Visualizations saved in: {VIZ_DIR}")
    print("="*60)
