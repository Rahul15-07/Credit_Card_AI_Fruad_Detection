import numpy as np
import pandas as pd
import streamlit as st

class FraudPredictor:
    def __init__(self, model_trainer, data_processor):
        self.model_trainer = model_trainer
        self.data_processor = data_processor
        self.risk_thresholds = {
            'very_low': 0.2,
            'low': 0.4,
            'medium': 0.6,
            'high': 0.8
        }
    
    def predict_single_transaction(self, transaction_data, model_name):
        """Predict fraud for a single transaction with detailed risk assessment"""
        # Ensure transaction data is properly formatted
        if isinstance(transaction_data, dict):
            # Convert to DataFrame
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        # Scale the data using the same scaler used during training
        scaled_data = self.data_processor.scaler.transform(df)
        
        # Get the model
        if model_name not in self.model_trainer.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.model_trainer.models[model_name]
        
        # Make prediction
        if model_name == 'Neural Network':
            fraud_probability = model.predict(scaled_data)[0][0]
            prediction = int(fraud_probability > 0.5)
        else:
            prediction = model.predict(scaled_data)[0]
            fraud_probability = model.predict_proba(scaled_data)[0][1]
        
        # Calculate risk assessment
        risk_assessment = self._assess_risk(fraud_probability)
        
        return {
            'prediction': prediction,
            'fraud_probability': fraud_probability,
            'legitimate_probability': 1 - fraud_probability,
            'risk_level': risk_assessment['level'],
            'risk_color': risk_assessment['color'],
            'recommendation': risk_assessment['recommendation'],
            'confidence': max(fraud_probability, 1 - fraud_probability)
        }
    
    def batch_predict(self, transactions_data, model_name):
        """Predict fraud for multiple transactions"""
        results = []
        
        for i, transaction in transactions_data.iterrows():
            # Remove target column if present
            transaction_features = transaction.drop('is_fraud', errors='ignore')
            result = self.predict_single_transaction(transaction_features, model_name)
            result['transaction_id'] = i
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _assess_risk(self, fraud_probability):
        """Assess risk level based on fraud probability"""
        if fraud_probability >= self.risk_thresholds['high']:
            return {
                'level': 'Very High',
                'color': '🔴',
                'recommendation': 'Block transaction immediately and contact customer'
            }
        elif fraud_probability >= self.risk_thresholds['medium']:
            return {
                'level': 'High',
                'color': '🟠',
                'recommendation': 'Hold transaction for manual review'
            }
        elif fraud_probability >= self.risk_thresholds['low']:
            return {
                'level': 'Medium',
                'color': '🟡',
                'recommendation': 'Monitor transaction patterns closely'
            }
        elif fraud_probability >= self.risk_thresholds['very_low']:
            return {
                'level': 'Low',
                'color': '🟢',
                'recommendation': 'Proceed with standard verification'
            }
        else:
            return {
                'level': 'Very Low',
                'color': '🟢',
                'recommendation': 'Process transaction normally'
            }
    
    def analyze_transaction_patterns(self, transaction_data):
        """Analyze patterns in transaction data"""
        analysis = {}
        
        # Feature statistics
        for feature in self.data_processor.feature_names:
            if feature in transaction_data:
                feature_value = transaction_data[feature]
                feature_stats = self.data_processor.data[feature].describe()
                
                # Calculate percentile
                percentile = (self.data_processor.data[feature] <= feature_value).mean() * 100
                
                analysis[feature] = {
                    'value': feature_value,
                    'percentile': percentile,
                    'is_outlier': percentile < 5 or percentile > 95,
                    'mean': feature_stats['mean'],
                    'std': feature_stats['std']
                }
        
        return analysis
    
    def get_fraud_indicators(self, transaction_data, model_name):
        """Get specific indicators that suggest fraud"""
        # Get feature analysis
        pattern_analysis = self.analyze_transaction_patterns(transaction_data)
        
        # Get model-specific insights
        indicators = []
        
        # Check for outliers
        for feature, stats in pattern_analysis.items():
            if stats['is_outlier']:
                if stats['percentile'] < 5:
                    indicators.append(f"⚠️ {feature} is unusually low (bottom 5%)")
                elif stats['percentile'] > 95:
                    indicators.append(f"⚠️ {feature} is unusually high (top 5%)")
        
        # Get feature importance if available
        if hasattr(self.model_trainer, 'rf_model') and self.model_trainer.rf_model is not None:
            feature_importance = self.model_trainer.get_feature_importance('Random Forest', self.data_processor.feature_names)
            
            if feature_importance is not None:
                # Check top important features
                top_features = feature_importance.head(5)['feature'].tolist()
                
                for feature in top_features:
                    if feature in pattern_analysis and pattern_analysis[feature]['is_outlier']:
                        indicators.append(f"🚨 High-importance feature {feature} shows unusual pattern")
        
        return indicators
    
    def simulate_transaction_stream(self, num_transactions=10):
        """Simulate a stream of transactions for real-time testing"""
        # Generate random transactions based on the original data distribution
        simulated_transactions = []
        
        for _ in range(num_transactions):
            transaction = {}
            
            for feature in self.data_processor.feature_names:
                # Get feature statistics from original data
                feature_stats = self.data_processor.data[feature].describe()
                
                # Generate random value within realistic range
                min_val = feature_stats['min']
                max_val = feature_stats['max']
                mean_val = feature_stats['mean']
                std_val = feature_stats['std']
                
                # Use normal distribution around mean with some randomness
                value = np.random.normal(mean_val, std_val * 0.5)
                value = np.clip(value, min_val, max_val)
                
                transaction[feature] = value
            
            simulated_transactions.append(transaction)
        
        return pd.DataFrame(simulated_transactions)
