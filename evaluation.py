"""
Evaluation metrics and visualization utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Dict, Tuple

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: list = ['Fake', 'Real'],
                         save_path: str = 'confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_training_history(history: Dict, save_path: str = 'training_history.png'):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Train Accuracy', marker='o')
    if 'val_accuracy' in history:
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        axes[1].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")

def plot_sample_images(images: np.ndarray, labels: np.ndarray, 
                      predictions: np.ndarray = None,
                      n_samples: int = 16,
                      save_path: str = 'sample_images.png'):
    """
    Plot sample images with their labels and predictions.
    
    Args:
        images: Image array
        labels: True labels
        predictions: Predicted labels (optional)
        n_samples: Number of samples to display
        save_path: Path to save the plot
    """
    n_samples = min(n_samples, len(images))
    indices = np.random.choice(len(images), n_samples, replace=False)
    
    cols = 4
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for idx, ax in enumerate(axes):
        if idx < n_samples:
            img_idx = indices[idx]
            ax.imshow(images[img_idx])
            title = f"True: {'Real' if labels[img_idx] == 1 else 'Fake'}"
            if predictions is not None:
                title += f"\nPred: {'Real' if predictions[img_idx] == 1 else 'Fake'}"
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample images saved to {save_path}")

def plot_metrics_comparison(cnn_metrics: Dict, fsl_metrics: Dict,
                           save_path: str = 'metrics_comparison.png'):
    """
    Plot comparison of metrics between CNN and FSL models.
    
    Args:
        cnn_metrics: Metrics from CNN model
        fsl_metrics: Metrics from FSL model
        save_path: Path to save the plot
    """
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    cnn_values = [cnn_metrics['accuracy'], cnn_metrics['precision'], 
                  cnn_metrics['recall'], cnn_metrics['f1_score']]
    fsl_values = [fsl_metrics['accuracy'], fsl_metrics['precision'], 
                  fsl_metrics['recall'], fsl_metrics['f1_score']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, cnn_values, width, label='CNN', alpha=0.8)
    bars2 = ax.bar(x + width/2, fsl_values, width, label='Proposed System (FSL)', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison saved to {save_path}")

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: list = ['Fake', 'Real']):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("="*60 + "\n")

def save_results_table(results: Dict, save_path: str = 'results_table.txt'):
    """
    Save results table to file.
    
    Args:
        results: Dictionary of results
        save_path: Path to save the file
    """
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EXPERIMENTAL RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for method, metrics in results.items():
            f.write(f"{method}:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score:  {metrics['f1_score']:.4f}\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
    
    print(f"Results table saved to {save_path}")

