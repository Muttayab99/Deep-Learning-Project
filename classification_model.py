"""
CNN and Few-Shot Learning (FSL) models for image classification.
Based on the paper's architecture for rare data classification.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CNNFeatureExtractor:
    """CNN model for feature extraction."""
    
    def __init__(self, img_size: tuple = (224, 224, 3)):
        """
        Initialize the CNN feature extractor.
        
        Args:
            img_size: Input image size (height, width, channels)
        """
        self.img_size = img_size
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build the CNN architecture as described in the paper."""
        inputs = layers.Input(shape=self.img_size)
        
        # First block: 32 filters, 3x3
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Second block: 64 filters, 3x3
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Third block: 128 filters, 3x3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Flatten
        x = layers.Flatten()(x)
        
        # Dense layer for feature extraction
        features = layers.Dense(512, activation='relu', name='features')(x)
        features_dropout = layers.Dropout(0.5, name='features_dropout')(features)
        
        # Binary classification output
        outputs = layers.Dense(1, activation='sigmoid', name='classification')(features_dropout)
        
        model = keras.Model(inputs, outputs, name='cnn_feature_extractor')
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: Input images (batch_size, H, W, C)
            
        Returns:
            Feature vectors (batch_size, feature_dim)
        """
        feature_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('features').output
        )
        return feature_model.predict(images, verbose=0)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32):
        """
        Train the CNN model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                'best_cnn_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

class FewShotLearningClassifier:
    """Few-Shot Learning classifier using cosine similarity."""
    
    def __init__(self, feature_extractor: CNNFeatureExtractor):
        """
        Initialize the FSL classifier.
        
        Args:
            feature_extractor: Trained CNN feature extractor
        """
        self.feature_extractor = feature_extractor
        self.prototypes = {}  # Class prototypes (mean feature vectors)
    
    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def compute_prototypes(self, support_images: np.ndarray, support_labels: np.ndarray):
        """
        Compute class prototypes from support set.
        
        Args:
            support_images: Support set images
            support_labels: Support set labels (0 or 1)
        """
        # Extract features
        features = self.feature_extractor.extract_features(support_images)
        
        # Compute mean feature vector for each class
        unique_labels = np.unique(support_labels)
        self.prototypes = {}
        
        for label in unique_labels:
            mask = support_labels == label
            if np.sum(mask) == 0:
                continue
            class_features = features[mask]
            prototype = np.mean(class_features, axis=0)
            self.prototypes[int(label)] = prototype
        
        # Ensure we have both classes (0 and 1) for binary classification
        if 0 not in self.prototypes and 1 in self.prototypes:
            # Use a zero vector or negative of class 1 as fallback
            self.prototypes[0] = -self.prototypes[1]
        elif 1 not in self.prototypes and 0 in self.prototypes:
            # Use a zero vector or negative of class 0 as fallback
            self.prototypes[1] = -self.prototypes[0]
        
        print(f"Computed prototypes for classes: {list(self.prototypes.keys())}")
    
    def predict(self, query_images: np.ndarray) -> np.ndarray:
        """
        Predict labels for query images using cosine similarity.
        
        Args:
            query_images: Query images to classify
            
        Returns:
            Predicted labels (0 or 1)
        """
        if not self.prototypes:
            raise ValueError("Prototypes not computed. Call compute_prototypes first.")
        
        # Extract features
        query_features = self.feature_extractor.extract_features(query_images)
        
        # Compute similarities to each prototype
        predictions = []
        similarities = []
        
        for query_feat in query_features:
            class_similarities = {}
            for class_label, prototype in self.prototypes.items():
                similarity = self.compute_cosine_similarity(query_feat, prototype)
                class_similarities[class_label] = similarity
            
            # Predict class with highest similarity
            predicted_class = max(class_similarities, key=class_similarities.get)
            predictions.append(predicted_class)
            similarities.append(class_similarities)
        
        return np.array(predictions), similarities
    
    def predict_proba(self, query_images: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for query images.
        
        Args:
            query_images: Query images to classify
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.prototypes:
            raise ValueError("Prototypes not computed. Call compute_prototypes first.")
        
        # Extract features
        query_features = self.feature_extractor.extract_features(query_images)
        
        probabilities = []
        class_labels = sorted(self.prototypes.keys())
        
        for query_feat in query_features:
            class_similarities = []
            for class_label in class_labels:
                similarity = self.compute_cosine_similarity(query_feat, self.prototypes[class_label])
                class_similarities.append(similarity)
            
            # Convert similarities to probabilities using softmax
            similarities = np.array(class_similarities)
            # Normalize to [0, 1] range and apply softmax
            exp_sim = np.exp(similarities - np.max(similarities))
            probs = exp_sim / np.sum(exp_sim)
            probabilities.append(probs)
        
        return np.array(probabilities)

