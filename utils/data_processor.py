import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import streamlit as st

class DataProcessor:
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load_data(self, df):
        """Load and validate the dataset"""
        self.data = df.copy()
        self.feature_names = [col for col in df.columns if col != 'is_fraud']
        
        # Basic validation
        if 'is_fraud' not in df.columns:
            raise ValueError("Dataset must contain 'is_fraud' column")
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            st.warning("Dataset contains missing values. Consider data cleaning.")
        
        return True
    
    def preprocess_data(self, test_size=0.2, use_smote=True, random_state=42):
        """
        Preprocess the data for training
        - Split into train/test sets
        - Scale features
        - Apply SMOTE if requested
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Separate features and target
        X = self.data[self.feature_names]
        y = self.data['is_fraud']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE if requested
        if use_smote:
            smote = SMOTE(random_state=random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            
            # Log the resampling effect
            original_fraud_count = sum(y_train)
            st.info(f"SMOTE applied: Fraud cases increased from original count")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_data_summary(self):
        """Get summary statistics of the dataset"""
        if self.data is None:
            return None
        
        summary = {
            'total_transactions': len(self.data),
            'fraud_cases': sum(self.data['is_fraud']),
            'fraud_rate': sum(self.data['is_fraud']) / len(self.data) * 100,
            'features': len(self.feature_names),
            'missing_values': self.data.isnull().sum().sum()
        }
        
        return summary
    
    def get_feature_statistics(self):
        """Get detailed statistics for all features"""
        if self.data is None:
            return None
        
        # Basic statistics
        stats = self.data[self.feature_names].describe()
        
        # Add fraud correlation
        correlations = []
        for feature in self.feature_names:
            corr = self.data[feature].corr(self.data['is_fraud'])
            correlations.append(corr)
        
        stats.loc['fraud_correlation'] = correlations
        
        return stats
    
    def detect_outliers(self, method='iqr'):
        """Detect outliers in the dataset"""
        if self.data is None:
            return None
        
        outliers_info = {}
        
        for feature in self.feature_names:
            if method == 'iqr':
                Q1 = self.data[feature].quantile(0.25)
                Q3 = self.data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.data[
                    (self.data[feature] < lower_bound) | 
                    (self.data[feature] > upper_bound)
                ]
                
                outliers_info[feature] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(self.data) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        return outliers_info
    
    def get_class_balance_info(self):
        """Get information about class balance"""
        if self.data is None:
            return None
        
        fraud_count = sum(self.data['is_fraud'])
        legitimate_count = len(self.data) - fraud_count
        
        balance_info = {
            'fraud_cases': fraud_count,
            'legitimate_cases': legitimate_count,
            'fraud_percentage': fraud_count / len(self.data) * 100,
            'legitimate_percentage': legitimate_count / len(self.data) * 100,
            'imbalance_ratio': legitimate_count / fraud_count if fraud_count > 0 else float('inf')
        }
        
        return balance_info
