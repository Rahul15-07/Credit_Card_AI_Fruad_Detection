import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import streamlit as st

class EnsembleModel:
    def __init__(self):
        self.voting_model = None
        self.stacking_model = None
        self.weighted_model = None
        self.ensemble_results = {}
        
    def create_voting_ensemble(self, models, voting_type='soft'):
        """
        Create a voting ensemble from multiple models
        
        Parameters:
        - models: dict of trained models {'name': model}
        - voting_type: 'soft' (probability-based) or 'hard' (majority vote)
        """
        estimators = [(name, model) for name, model in models.items()]
        
        self.voting_model = VotingClassifier(
            estimators=estimators,
            voting=voting_type
        )
        
        return self.voting_model
    
    def create_stacking_ensemble(self, models, meta_learner=None):
        """
        Create a stacking ensemble with a meta-learner
        
        Parameters:
        - models: dict of base models
        - meta_learner: meta-classifier (default: LogisticRegression)
        """
        if meta_learner is None:
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        estimators = [(name, model) for name, model in models.items()]
        
        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5
        )
        
        return self.stacking_model
    
    def create_weighted_average_ensemble(self, models, weights=None):
        """
        Create a weighted average ensemble
        
        Parameters:
        - models: dict of trained models
        - weights: dict of model weights (default: equal weights)
        """
        if weights is None:
            # Equal weights for all models
            weights = {name: 1/len(models) for name in models.keys()}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        self.weighted_model = {
            'models': models,
            'weights': normalized_weights
        }
        
        return self.weighted_model
    
    def train_voting_ensemble(self, X_train, X_test, y_train, y_test, models, voting_type='soft'):
        """Train voting ensemble and evaluate"""
        self.create_voting_ensemble(models, voting_type)
        
        # Train the ensemble
        self.voting_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.voting_model.predict(X_test)
        y_pred_proba = self.voting_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        self.ensemble_results['Voting Ensemble'] = metrics
        
        return metrics
    
    def train_stacking_ensemble(self, X_train, X_test, y_train, y_test, models, meta_learner=None):
        """Train stacking ensemble and evaluate"""
        self.create_stacking_ensemble(models, meta_learner)
        
        # Train the ensemble
        self.stacking_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.stacking_model.predict(X_test)
        y_pred_proba = self.stacking_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        self.ensemble_results['Stacking Ensemble'] = metrics
        
        return metrics
    
    def train_weighted_ensemble(self, X_test, y_test, models, weights=None):
        """Train weighted average ensemble and evaluate"""
        self.create_weighted_average_ensemble(models, weights)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities[name] = model.predict_proba(X_test)[:, 1]
                else:
                    # For neural networks or other models
                    pred = model.predict(X_test, verbose=0) if hasattr(model, 'predict') else model.predict(X_test)
                    probabilities[name] = pred.flatten() if hasattr(pred, 'flatten') else pred
            except Exception as e:
                print(f"Warning: Could not get predictions from {name}: {str(e)}")
                continue
        
        # Calculate weighted average of probabilities
        weighted_proba = np.zeros(len(X_test))
        for name, weight in self.weighted_model['weights'].items():
            weighted_proba += weight * probabilities[name]
        
        # Convert to predictions
        y_pred = (weighted_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, weighted_proba)
        
        self.ensemble_results['Weighted Average Ensemble'] = metrics
        
        return metrics
    
    def predict_voting(self, X):
        """Make predictions using voting ensemble"""
        if self.voting_model is None:
            raise ValueError("Voting ensemble not trained")
        
        prediction = self.voting_model.predict(X)
        probability = self.voting_model.predict_proba(X)[:, 1]
        
        return prediction, probability
    
    def predict_stacking(self, X):
        """Make predictions using stacking ensemble"""
        if self.stacking_model is None:
            raise ValueError("Stacking ensemble not trained")
        
        prediction = self.stacking_model.predict(X)
        probability = self.stacking_model.predict_proba(X)[:, 1]
        
        return prediction, probability
    
    def predict_weighted(self, X):
        """Make predictions using weighted average ensemble"""
        if self.weighted_model is None:
            raise ValueError("Weighted ensemble not created")
        
        # Get predictions from all models
        probabilities = {}
        
        for name, model in self.weighted_model['models'].items():
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities[name] = model.predict_proba(X)[:, 1]
                else:
                    # For neural networks or other models
                    pred = model.predict(X, verbose=0) if hasattr(model, 'predict') else model.predict(X)
                    probabilities[name] = pred.flatten() if hasattr(pred, 'flatten') else pred
            except Exception as e:
                print(f"Warning: Could not get predictions from {name}: {str(e)}")
                continue
        
        # Calculate weighted average
        weighted_proba = np.zeros(len(X))
        for name, weight in self.weighted_model['weights'].items():
            weighted_proba += weight * probabilities[name]
        
        prediction = (weighted_proba > 0.5).astype(int)
        
        return prediction, weighted_proba
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
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
    
    def optimize_weights(self, X_val, y_val, models, search_space=10):
        """
        Optimize weights for weighted ensemble using grid search
        
        Parameters:
        - X_val: validation features
        - y_val: validation labels
        - models: dict of models
        - search_space: number of weight combinations to try
        """
        best_weights = None
        best_score = 0
        
        # Generate weight combinations
        model_names = list(models.keys())
        
        for _ in range(search_space * 10):
            # Random weights
            weights = np.random.dirichlet(np.ones(len(models)))
            weight_dict = {name: w for name, w in zip(model_names, weights)}
            
            # Calculate performance
            probabilities = {}
            for name, model in models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        probabilities[name] = model.predict_proba(X_val)[:, 1]
                    else:
                        pred = model.predict(X_val, verbose=0) if hasattr(model, 'predict') else model.predict(X_val)
                        probabilities[name] = pred.flatten() if hasattr(pred, 'flatten') else pred
                except Exception as e:
                    print(f"Warning: Could not get predictions from {name}: {str(e)}")
                    continue
            
            weighted_proba = np.zeros(len(X_val))
            for name, weight in weight_dict.items():
                weighted_proba += weight * probabilities[name]
            
            # Calculate AUC-ROC
            score = roc_auc_score(y_val, weighted_proba)
            
            if score > best_score:
                best_score = score
                best_weights = weight_dict
        
        return best_weights, best_score
    
    def get_ensemble_comparison(self):
        """Get comparison of all ensemble methods"""
        if not self.ensemble_results:
            return None
        
        comparison = pd.DataFrame({
            model: {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc']
            }
            for model, metrics in self.ensemble_results.items()
        }).T
        
        return comparison
    
    def get_model_contributions(self, X_test, y_test, models):
        """
        Analyze individual model contributions to ensemble performance
        """
        contributions = {}
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    pred = model.predict(X_test, verbose=0) if hasattr(model, 'predict') else model.predict(X_test)
                    y_pred_proba = pred.flatten() if hasattr(pred, 'flatten') else pred
                
                auc = roc_auc_score(y_test, y_pred_proba)
                contributions[name] = auc
            except Exception as e:
                print(f"Warning: Could not evaluate {name}: {str(e)}")
                continue
        
        return contributions
