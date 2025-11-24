"""
Main script for reproducing the Rare Data Image Classification System Using Few-Shot Learning.
This script implements the complete pipeline as described in the research paper.
"""
import os
import numpy as np
import tensorflow as tf
from data_loader import PlantDataLoader
from data_augmentation import DataAugmentationModel
from classification_model import CNNFeatureExtractor, FewShotLearningClassifier
from evaluation import (calculate_metrics, plot_confusion_matrix, plot_training_history,
                       plot_sample_images, plot_metrics_comparison, print_classification_report,
                       save_results_table)
import random

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_fake_images_with_augmentation(data_loader: PlantDataLoader, 
                                         real_images: np.ndarray,
                                         n_fake: int = None) -> np.ndarray:
    """
    Create fake images using data augmentation model.
    
    Args:
        data_loader: Data loader instance
        real_images: Real images to use as source
        n_fake: Number of fake images to generate (None = same as real)
        
    Returns:
        Generated fake images
    """
    print("\n" + "="*60)
    print("CREATING FAKE IMAGES USING DATA AUGMENTATION")
    print("="*60)
    
    n_fake = n_fake or len(real_images)
    
    # Initialize augmentation model
    aug_model = DataAugmentationModel(img_size=(224, 224, 3))
    aug_model.compile_model(learning_rate=0.001)
    
    # Get image pairs for training
    print("Preparing image pairs for augmentation model training...")
    pairs = data_loader.get_image_pairs(min(100, len(real_images)))
    source_imgs = np.array([p[0] for p in pairs])
    ref_imgs = np.array([p[1] for p in pairs])
    
    # Train augmentation model (quick training for demonstration)
    print("Training data augmentation model...")
    aug_model.train(source_imgs, ref_imgs, epochs=8, batch_size=8, validation_split=0.2)
    
    # Generate fake images
    print(f"Generating {n_fake} fake images...")
    fake_images = []
    
    for i in range(n_fake):
        # Randomly select source and reference images
        source_idx = np.random.randint(0, len(real_images))
        ref_idx = np.random.randint(0, len(real_images))
        
        source_img = real_images[source_idx]
        ref_img = real_images[ref_idx]
        
        # Generate augmented image
        fake_img = aug_model.generate_augmented_image(source_img, ref_img)
        fake_images.append(fake_img)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{n_fake} fake images...")
    
    fake_images = np.array(fake_images)
    print(f"Generated {len(fake_images)} fake images successfully!")
    
    return fake_images

def prepare_dataset(real_images: np.ndarray, fake_images: np.ndarray,
                   test_size: float = 0.2) -> tuple:
    """
    Prepare train/test split for the dataset.
    
    Args:
        real_images: Real images
        fake_images: Fake images
        test_size: Test set proportion
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Combine real and fake images
    all_images = np.concatenate([real_images, fake_images], axis=0)
    all_labels = np.concatenate([
        np.ones(len(real_images)),  # Real = 1
        np.zeros(len(fake_images))   # Fake = 0
    ], axis=0)
    
    # Shuffle
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    # Split
    split_idx = int(len(all_images) * (1 - test_size))
    X_train = all_images[:split_idx]
    y_train = all_labels[:split_idx]
    X_test = all_images[split_idx:]
    y_test = all_labels[split_idx:]
    
    print(f"\nDataset prepared:")
    print(f"  Train: {len(X_train)} samples ({np.sum(y_train==1)} real, {np.sum(y_train==0)} fake)")
    print(f"  Test:  {len(X_test)} samples ({np.sum(y_test==1)} real, {np.sum(y_test==0)} fake)")
    
    return X_train, X_test, y_train, y_test

def train_cnn_model(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    epochs: int = 100, batch_size: int = 32) -> tuple:
    """
    Train CNN model.
    
    Args:
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Trained model and history
    """
    print("\n" + "="*60)
    print("TRAINING CNN MODEL")
    print("="*60)
    
    cnn_model = CNNFeatureExtractor(img_size=(224, 224, 3))
    cnn_model.compile_model(learning_rate=0.001)
    
    history = cnn_model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    
    return cnn_model, history.history

def train_fsl_model(cnn_model: CNNFeatureExtractor,
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> FewShotLearningClassifier:
    """
    Train FSL model using the CNN feature extractor.
    
    Args:
        cnn_model: Trained CNN model
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        
    Returns:
        Trained FSL classifier
    """
    print("\n" + "="*60)
    print("TRAINING FEW-SHOT LEARNING MODEL")
    print("="*60)
    
    # Create FSL classifier
    fsl_classifier = FewShotLearningClassifier(cnn_model)
    
    # Compute prototypes from training data
    print("Computing class prototypes from training data...")
    fsl_classifier.compute_prototypes(X_train, y_train)
    
    return fsl_classifier

def evaluate_models(cnn_model: CNNFeatureExtractor,
                   fsl_classifier: FewShotLearningClassifier,
                   X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate both CNN and FSL models.
    
    Args:
        cnn_model: Trained CNN model
        fsl_classifier: Trained FSL classifier
        X_test: Test images
        y_test: Test labels
        
    Returns:
        Dictionary of results
    """
    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)
    
    results = {}
    
    # Evaluate CNN
    print("\nEvaluating CNN model...")
    cnn_predictions = cnn_model.model.predict(X_test, verbose=0)
    cnn_pred_labels = (cnn_predictions > 0.5).astype(int).flatten()
    cnn_metrics = calculate_metrics(y_test, cnn_pred_labels)
    results['CNN'] = cnn_metrics
    
    print(f"CNN Results:")
    print(f"  Accuracy:  {cnn_metrics['accuracy']:.4f}")
    print(f"  Precision: {cnn_metrics['precision']:.4f}")
    print(f"  Recall:    {cnn_metrics['recall']:.4f}")
    print(f"  F1 Score:  {cnn_metrics['f1_score']:.4f}")
    
    # Evaluate FSL
    print("\nEvaluating FSL model...")
    fsl_pred_labels, _ = fsl_classifier.predict(X_test)
    fsl_metrics = calculate_metrics(y_test, fsl_pred_labels)
    results['Proposed System (FSL)'] = fsl_metrics
    
    print(f"FSL Results:")
    print(f"  Accuracy:  {fsl_metrics['accuracy']:.4f}")
    print(f"  Precision: {fsl_metrics['precision']:.4f}")
    print(f"  Recall:    {fsl_metrics['recall']:.4f}")
    print(f"  F1 Score:  {fsl_metrics['f1_score']:.4f}")
    
    # Calculate improvement
    improvement = {
        'accuracy': fsl_metrics['accuracy'] - cnn_metrics['accuracy'],
        'precision': fsl_metrics['precision'] - cnn_metrics['precision'],
        'recall': fsl_metrics['recall'] - cnn_metrics['recall'],
        'f1_score': fsl_metrics['f1_score'] - cnn_metrics['f1_score']
    }
    
    print(f"\nImprovement over CNN:")
    print(f"  Accuracy:  {improvement['accuracy']:+.4f} ({improvement['accuracy']*100:+.2f}%)")
    print(f"  Precision: {improvement['precision']:+.4f} ({improvement['precision']*100:+.2f}%)")
    print(f"  Recall:    {improvement['recall']:+.4f} ({improvement['recall']*100:+.2f}%)")
    print(f"  F1 Score:  {improvement['f1_score']:+.4f} ({improvement['f1_score']*100:+.2f}%)")
    
    return results, cnn_pred_labels, fsl_pred_labels

def main():
    """Main execution function."""
    print("="*60)
    print("RARE DATA IMAGE CLASSIFICATION SYSTEM")
    print("Using Few-Shot Learning")
    print("="*60)
    
    # Configuration
    DATA_DIR = "plant source image"
    MAX_SAMPLES = 100  # Use 100 samples as in the paper
    IMG_SIZE = (224, 224)
    EPOCHS = 8  # Reduced for faster execution
    BATCH_SIZE = 32
    
    # Create output directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    data_loader = PlantDataLoader(DATA_DIR, img_size=IMG_SIZE)
    X_real_train, X_real_test, _, _ = data_loader.load_dataset(
        max_samples=MAX_SAMPLES, test_size=0.2
    )
    
    # Combine train and test for full dataset
    X_real_all = np.concatenate([X_real_train, X_real_test], axis=0)
    
    # Create fake images
    X_fake_all = create_fake_images_with_augmentation(data_loader, X_real_all, n_fake=len(X_real_all))
    
    # Prepare dataset
    X_train, X_test, y_train, y_test = prepare_dataset(X_real_all, X_fake_all, test_size=0.2)
    
    # Split validation set
    # Ensure we have enough data
    if len(X_train) < 20:
        print("Warning: Very small dataset. Adjusting train/val split...")
        val_split = max(1, len(X_train) // 5)
    else:
        val_split = int(len(X_train) * 0.2)
    
    X_val = X_train[:val_split]
    y_val = y_train[:val_split]
    X_train = X_train[val_split:]
    y_train = y_train[val_split:]
    
    # Train CNN model
    cnn_model, cnn_history = train_cnn_model(
        X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    
    # Train FSL model
    fsl_classifier = train_fsl_model(cnn_model, X_train, y_train, X_val, y_val)
    
    # Evaluate models
    results, cnn_pred, fsl_pred = evaluate_models(cnn_model, fsl_classifier, X_test, y_test)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Training history
    plot_training_history(cnn_history, 'plots/training_history.png')
    
    # Confusion matrices
    plot_confusion_matrix(y_test, cnn_pred, save_path='plots/confusion_matrix_cnn.png')
    plot_confusion_matrix(y_test, fsl_pred, save_path='plots/confusion_matrix_fsl.png')
    
    # Sample images
    plot_sample_images(X_test, y_test, cnn_pred, save_path='plots/sample_images_cnn.png')
    plot_sample_images(X_test, y_test, fsl_pred, save_path='plots/sample_images_fsl.png')
    
    # Metrics comparison
    plot_metrics_comparison(results['CNN'], results['Proposed System (FSL)'], 
                           save_path='plots/metrics_comparison.png')
    
    # Classification reports
    print("\nCNN Classification Report:")
    print_classification_report(y_test, cnn_pred)
    
    print("\nFSL Classification Report:")
    print_classification_report(y_test, fsl_pred)
    
    # Save results
    save_results_table(results, 'results/results_table.txt')
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nResults saved in 'results/' directory")
    print("Visualizations saved in 'plots/' directory")
    print("\nFinal Results Summary:")
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()

