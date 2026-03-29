import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = None
        self.autoencoder = None
        self.encoder = None
        self.threshold = None
        self.results = {}
        
    def train_isolation_forest(self, X_train, X_test, y_train, y_test, contamination=0.1, n_estimators=100):
        """
        Train Isolation Forest for anomaly detection
        
        Parameters:
        - contamination: Expected proportion of outliers (fraud rate)
        - n_estimators: Number of trees
        """
        # Initialize Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit on training data (legitimate transactions primarily)
        self.isolation_forest.fit(X_train)
        
        # Predict anomaly scores (-1 for outliers, 1 for inliers)
        train_scores = self.isolation_forest.decision_function(X_train)
        test_scores = self.isolation_forest.decision_function(X_test)
        
        # Convert to probabilities (higher score = more normal)
        # Normalize scores to [0, 1] range where 1 = fraud
        train_proba = self._normalize_scores(train_scores)
        test_proba = self._normalize_scores(test_scores)
        
        # Predictions (-1 = fraud, 1 = normal)
        train_pred = self.isolation_forest.predict(X_train)
        test_pred = self.isolation_forest.predict(X_test)
        
        # Convert to binary (0 = normal, 1 = fraud)
        train_pred_binary = np.where(train_pred == -1, 1, 0)
        test_pred_binary = np.where(test_pred == -1, 1, 0)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, test_pred_binary, test_proba)
        metrics['train_metrics'] = self._calculate_metrics(y_train, train_pred_binary, train_proba)
        
        self.results['Isolation Forest'] = metrics
        
        return metrics
    
    def train_autoencoder(self, X_train, X_test, y_train, y_test, 
                         encoding_dim=10, epochs=50, batch_size=32):
        """
        Train Autoencoder for anomaly detection
        
        Parameters:
        - encoding_dim: Dimension of encoded representation
        - epochs: Training epochs
        - batch_size: Batch size for training
        """
        input_dim = X_train.shape[1]
        
        # Build encoder
        encoder_input = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(encoder_input)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        self.encoder = keras.Model(encoder_input, encoded, name='encoder')
        
        # Build decoder
        decoder_input = keras.Input(shape=(encoding_dim,))
        decoded = layers.Dense(32, activation='relu')(decoder_input)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        decoder = keras.Model(decoder_input, decoded, name='decoder')
        
        # Build autoencoder
        autoencoder_input = keras.Input(shape=(input_dim,))
        encoded_repr = self.encoder(autoencoder_input)
        decoded_output = decoder(encoded_repr)
        
        self.autoencoder = keras.Model(autoencoder_input, decoded_output, name='autoencoder')
        
        # Compile
        self.autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train only on normal transactions (fraud=0)
        # This helps the autoencoder learn normal patterns
        normal_indices = np.where(y_train == 0)[0]
        X_train_normal = X_train[normal_indices]
        
        # Train autoencoder
        history = self.autoencoder.fit(
            X_train_normal, X_train_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        # Calculate reconstruction errors
        train_reconstructions = self.autoencoder.predict(X_train, verbose=0)
        test_reconstructions = self.autoencoder.predict(X_test, verbose=0)
        
        train_mse = np.mean(np.power(X_train - train_reconstructions, 2), axis=1)
        test_mse = np.mean(np.power(X_test - test_reconstructions, 2), axis=1)
        
        # Set threshold based on training data (e.g., 95th percentile)
        self.threshold = np.percentile(train_mse, 95)
        
        # Predictions (1 if error > threshold, 0 otherwise)
        train_pred = (train_mse > self.threshold).astype(int)
        test_pred = (test_mse > self.threshold).astype(int)
        
        # Use reconstruction error as probability score
        train_proba = self._normalize_reconstruction_error(train_mse)
        test_proba = self._normalize_reconstruction_error(test_mse)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, test_pred, test_proba)
        metrics['train_metrics'] = self._calculate_metrics(y_train, train_pred, train_proba)
        metrics['threshold'] = self.threshold
        metrics['history'] = history.history
        
        self.results['Autoencoder'] = metrics
        
        return metrics
    
    def predict_isolation_forest(self, X):
        """Predict using Isolation Forest"""
        if self.isolation_forest is None:
            raise ValueError("Isolation Forest not trained")
        
        scores = self.isolation_forest.decision_function(X)
        probabilities = self._normalize_scores(scores)
        predictions = self.isolation_forest.predict(X)
        predictions_binary = np.where(predictions == -1, 1, 0)
        
        return predictions_binary, probabilities
    
    def predict_autoencoder(self, X):
        """Predict using Autoencoder"""
        if self.autoencoder is None:
            raise ValueError("Autoencoder not trained")
        
        reconstructions = self.autoencoder.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        predictions = (mse > self.threshold).astype(int)
        probabilities = self._normalize_reconstruction_error(mse)
        
        return predictions, probabilities
    
    def _normalize_scores(self, scores):
        """Normalize isolation forest scores to [0, 1] range"""
        # Scores are typically negative, with more negative = more anomalous
        # Convert to probability: more negative = higher fraud probability
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.zeros_like(scores)
        
        # Normalize and invert (lower score = higher fraud probability)
        normalized = (scores - min_score) / (max_score - min_score)
        inverted = 1 - normalized
        
        return inverted
    
    def _normalize_reconstruction_error(self, errors):
        """Normalize reconstruction errors to [0, 1] range"""
        # Higher error = higher fraud probability
        if self.threshold is None or self.threshold == 0:
            # If no threshold, use percentile normalization
            max_error = np.percentile(errors, 99)
        else:
            max_error = self.threshold * 2
        
        normalized = np.clip(errors / max_error, 0, 1)
        
        return normalized
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'roc_data': roc_curve(y_true, y_pred_proba)
        }
        
        return metrics
    
    def get_reconstruction_error_distribution(self, X, y):
        """Get distribution of reconstruction errors for visualization"""
        if self.autoencoder is None:
            return None
        
        reconstructions = self.autoencoder.predict(X, verbose=0)
        errors = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # Separate by actual class
        normal_errors = errors[y == 0]
        fraud_errors = errors[y == 1]
        
        return {
            'normal_errors': normal_errors,
            'fraud_errors': fraud_errors,
            'threshold': self.threshold
        }
    
    def get_anomaly_scores_distribution(self, X, y):
        """Get distribution of anomaly scores for visualization"""
        if self.isolation_forest is None:
            return None
        
        scores = self.isolation_forest.decision_function(X)
        
        # Separate by actual class
        normal_scores = scores[y == 0]
        fraud_scores = scores[y == 1]
        
        return {
            'normal_scores': normal_scores,
            'fraud_scores': fraud_scores
        }
    
    def compare_methods(self):
        """Compare both anomaly detection methods"""
        if not self.results:
            return None
        
        comparison = pd.DataFrame({
            method: {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc']
            }
            for method, metrics in self.results.items()
        }).T
        
        return comparison
    
    def explain_anomaly(self, X, method='autoencoder'):
        """Provide explanation for why a transaction is flagged as anomaly"""
        if method == 'autoencoder' and self.autoencoder is not None:
            reconstructions = self.autoencoder.predict(X, verbose=0)
            errors = np.abs(X - reconstructions)
            
            # Get top features with highest reconstruction error
            feature_errors = np.mean(errors, axis=0)
            top_features = np.argsort(feature_errors)[::-1][:5]
            
            return {
                'reconstruction_error': np.mean(errors),
                'top_anomalous_features': top_features,
                'feature_errors': feature_errors
            }
        
        elif method == 'isolation_forest' and self.isolation_forest is not None:
            score = self.isolation_forest.decision_function(X)[0]
            prediction = self.isolation_forest.predict(X)[0]
            
            return {
                'anomaly_score': score,
                'is_anomaly': prediction == -1,
                'note': 'Isolation Forest uses tree-based approach, feature importance not directly available'
            }
        
        return None
