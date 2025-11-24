"""
Data loading and preprocessing utilities for the rare data image classification system.
"""
import os
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import random

class PlantDataLoader:
    """Load and preprocess plant images for the classification system."""
    
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing plant images
            img_size: Target image size (height, width)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.image_paths = []
        self._load_image_paths()
    
    def _load_image_paths(self):
        """Load all image file paths from the directory."""
        # First, try direct directory
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                    self.image_paths.append(os.path.join(self.data_dir, filename))
        
        # If no images found, try nested directory
        if len(self.image_paths) == 0:
            nested_dir = os.path.join(self.data_dir, os.path.basename(self.data_dir))
            if os.path.exists(nested_dir):
                for filename in os.listdir(nested_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
                        self.image_paths.append(os.path.join(nested_dir, filename))
        
        print(f"Loaded {len(self.image_paths)} images from {self.data_dir}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array (normalized to [0, 1])
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                # Try with PIL as fallback
                img = np.array(Image.open(image_path))
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            return np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.float32)
    
    def load_dataset(self, max_samples: int = None, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split the dataset into train and test sets.
        
        Args:
            max_samples: Maximum number of samples to use (None for all)
            test_size: Proportion of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test (all images labeled as real=1)
        """
        # Limit samples if specified
        paths = self.image_paths[:max_samples] if max_samples else self.image_paths
        
        # Load images
        images = []
        for path in paths:
            img = self.load_image(path)
            images.append(img)
        
        images = np.array(images)
        
        # All images are labeled as real (1)
        labels = np.ones(len(images), dtype=np.int32)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, shuffle=True
        )
        
        print(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_image_pairs(self, n_pairs: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get random pairs of images for data augmentation.
        
        Args:
            n_pairs: Number of image pairs to generate
            
        Returns:
            List of (source_image, reference_image) tuples
        """
        pairs = []
        for _ in range(n_pairs):
            # Randomly select two different images
            idx1, idx2 = random.sample(range(len(self.image_paths)), 2)
            img1 = self.load_image(self.image_paths[idx1])
            img2 = self.load_image(self.image_paths[idx2])
            pairs.append((img1, img2))
        
        return pairs

