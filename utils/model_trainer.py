import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st

class ModelTrainer:
    def __init__(self):
        self.lr_model = None
        self.rf_model = None
        self.nn_model = None
        self.models = {}
        self.training_history = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_unscaled = None
        self.X_test_unscaled = None
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression model"""
        # Store training data for ensemble methods
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Initialize model with balanced class weights
        self.lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        # Train the model
        self.lr_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.lr_model.predict(X_test)
        y_pred_proba = self.lr_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Store model and results
        self.models['Logistic Regression'] = self.lr_model
        
        return metrics
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        # Store training data for ensemble methods
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Initialize model with balanced class weights
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        self.rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.rf_model.predict(X_test)
        y_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Store model and results
        self.models['Random Forest'] = self.rf_model
        
        return metrics
    
    def train_neural_network(self, X_train, X_test, y_train, y_test):
        """Train Neural Network model"""
        # Store training data for ensemble methods
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Calculate class weights for imbalanced data
        class_weight = {
            0: len(y_train) / (2 * np.sum(y_train == 0)),
            1: len(y_train) / (2 * np.sum(y_train == 1))
        }
        
        # Build neural network architecture
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            class_weight=class_weight,
            verbose=0  # Silent training
        )
        
        # Store training history
        self.training_history['Neural Network'] = history.history
        
        # Make predictions
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Store model and results
        self.nn_model = model
        self.models['Neural Network'] = model
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'roc_data': roc_curve(y_true, y_pred_proba)
        }
        
        return metrics
    
    def get_model_comparison(self):
        """Get comparison of all trained models"""
        if not self.models:
            return None
        
        comparison = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                comparison[name] = {
                    'feature_importances': model.feature_importances_,
                    'type': 'tree_based'
                }
            elif hasattr(model, 'coef_'):
                comparison[name] = {
                    'coefficients': model.coef_[0],
                    'type': 'linear'
                }
            else:
                comparison[name] = {
                    'type': 'neural_network'
                }
        
        return comparison
    
    def predict_transaction(self, model_name, transaction_data):
        """Predict fraud for a single transaction"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        if model_name == 'Neural Network':
            probability = model.predict(transaction_data.reshape(1, -1))[0][0]
            prediction = int(probability > 0.5)
        else:
            prediction = model.predict(transaction_data.reshape(1, -1))[0]
            probability = model.predict_proba(transaction_data.reshape(1, -1))[0][1]
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': max(probability, 1 - probability)
        }
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_data = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_data
        
        return None
    
    def get_training_history(self, model_name):
        """Get training history for neural network"""
        if model_name in self.training_history:
            return self.training_history[model_name]
        return None
