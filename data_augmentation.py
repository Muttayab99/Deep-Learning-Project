"""
Data augmentation model using encoder-decoder architecture for generating synthetic images.
Based on the paper's methodology for combining source and reference images.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM)."""
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = (img1 * 255).astype(np.uint8)
        img2_gray = (img2 * 255).astype(np.uint8)
    
    return ssim(img1_gray, img2_gray, data_range=255)

class DataAugmentationModel:
    """Encoder-decoder model for data augmentation."""
    
    def __init__(self, img_size: tuple = (224, 224, 3)):
        """
        Initialize the data augmentation model.
        
        Args:
            img_size: Input image size (height, width, channels)
        """
        self.img_size = img_size
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build the encoder-decoder architecture."""
        # Input: concatenated source and reference images
        input_img = layers.Input(shape=(self.img_size[0], self.img_size[1], self.img_size[2] * 2))
        
        # Encoder
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Bottleneck
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Output: reconstructed image (same size as input single image)
        output = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        model = keras.Model(input_img, output, name='data_augmentation_model')
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, source_images: np.ndarray, reference_images: np.ndarray, 
              epochs: int = 50, batch_size: int = 16, validation_split: float = 0.2):
        """
        Train the data augmentation model.
        
        Args:
            source_images: Source images (target for reconstruction)
            reference_images: Reference images (for feature combination)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        # Concatenate source and reference images
        X = np.concatenate([source_images, reference_images], axis=-1)
        y = source_images  # Target is to reconstruct source image
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def generate_augmented_image(self, source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
        """
        Generate an augmented image by combining source and reference images.
        
        Args:
            source_img: Source image (shape: (H, W, 3))
            reference_img: Reference image (shape: (H, W, 3))
            
        Returns:
            Generated augmented image
        """
        # Prepare input
        combined = np.concatenate([source_img, reference_img], axis=-1)
        combined = np.expand_dims(combined, axis=0)
        
        # Generate
        generated = self.model.predict(combined, verbose=0)[0]
        
        return generated
    
    def generate_augmented_dataset(self, source_images: np.ndarray, 
                                   reference_images: np.ndarray,
                                   quality_threshold_psnr: float = 20.0,
                                   quality_threshold_ssim: float = 0.5) -> np.ndarray:
        """
        Generate augmented dataset with quality filtering.
        
        Args:
            source_images: Source images
            reference_images: Reference images
            quality_threshold_psnr: Minimum PSNR threshold
            quality_threshold_ssim: Minimum SSIM threshold
            
        Returns:
            Filtered augmented images
        """
        augmented_images = []
        
        for source, reference in zip(source_images, reference_images):
            # Generate augmented image
            augmented = self.generate_augmented_image(source, reference)
            
            # Calculate quality metrics
            psnr = calculate_psnr(source, augmented)
            ssim_val = calculate_ssim(source, augmented)
            
            # Filter by quality
            if psnr >= quality_threshold_psnr and ssim_val >= quality_threshold_ssim:
                augmented_images.append(augmented)
        
        return np.array(augmented_images)

