import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import os
import pickle

class Utils:
    @staticmethod
    def save_model(model, filename):
        """Save trained model to file"""
        try:
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
            return True
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            return False
    
    @staticmethod
    def load_model(filename):
        """Load trained model from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    model = pickle.load(file)
                return model
            else:
                st.warning(f"Model file {filename} not found")
                return None
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    @staticmethod
    def format_currency(amount):
        """Format number as currency"""
        return f"${amount:,.2f}"
    
    @staticmethod
    def format_percentage(value):
        """Format decimal as percentage"""
        return f"{value * 100:.2f}%"
    
    @staticmethod
    def calculate_business_impact(confusion_matrix, transaction_amounts=None):
        """Calculate business impact of fraud detection"""
        tn, fp, fn, tp = confusion_matrix.ravel()
        
        # Default transaction amounts if not provided
        if transaction_amounts is None:
            avg_fraud_amount = 1000  # Average fraud transaction amount
            avg_legit_amount = 200   # Average legitimate transaction amount
        else:
            avg_fraud_amount = transaction_amounts.get('fraud', 1000)
            avg_legit_amount = transaction_amounts.get('legitimate', 200)
        
        # Calculate costs
        false_positive_cost = fp * avg_legit_amount * 0.1  # Cost of blocking legitimate transactions
        false_negative_cost = fn * avg_fraud_amount        # Cost of missing fraud
        
        # Calculate savings
        true_positive_savings = tp * avg_fraud_amount      # Fraud prevented
        
        # Net benefit
        net_benefit = true_positive_savings - false_positive_cost - false_negative_cost
        
        return {
            'false_positive_cost': false_positive_cost,
            'false_negative_cost': false_negative_cost,
            'true_positive_savings': true_positive_savings,
            'net_benefit': net_benefit,
            'fraud_prevented': tp,
            'legitimate_blocked': fp,
            'fraud_missed': fn
        }
    
    @staticmethod
    def generate_transaction_report(predictions_df):
        """Generate summary report of predictions"""
        total_transactions = len(predictions_df)
        fraud_detected = sum(predictions_df['prediction'])
        avg_fraud_prob = predictions_df['fraud_probability'].mean()
        
        # Risk level distribution
        risk_levels = predictions_df['risk_level'].value_counts()
        
        report = {
            'total_transactions': total_transactions,
            'fraud_detected': fraud_detected,
            'fraud_rate': fraud_detected / total_transactions * 100,
            'avg_fraud_probability': avg_fraud_prob,
            'risk_distribution': risk_levels.to_dict()
        }
        
        return report
    
    @staticmethod
    def create_alert_system(fraud_probability, threshold=0.8):
        """Create alert system based on fraud probability"""
        if fraud_probability >= threshold:
            alert = {
                'level': 'CRITICAL',
                'message': 'High probability fraud detected - Immediate action required',
                'color': 'red',
                'actions': [
                    'Block transaction immediately',
                    'Contact customer for verification',
                    'Flag account for monitoring'
                ]
            }
        elif fraud_probability >= 0.6:
            alert = {
                'level': 'HIGH',
                'message': 'Potential fraud detected - Manual review required',
                'color': 'orange',
                'actions': [
                    'Hold transaction for review',
                    'Request additional verification',
                    'Monitor account activity'
                ]
            }
        elif fraud_probability >= 0.4:
            alert = {
                'level': 'MEDIUM',
                'message': 'Suspicious activity detected - Monitor closely',
                'color': 'yellow',
                'actions': [
                    'Monitor transaction patterns',
                    'Log for analysis',
                    'Consider additional checks'
                ]
            }
        else:
            alert = {
                'level': 'LOW',
                'message': 'Transaction appears legitimate',
                'color': 'green',
                'actions': [
                    'Process normally',
                    'Standard monitoring'
                ]
            }
        
        return alert
    
    @staticmethod
    def validate_transaction_data(transaction_data, required_features):
        """Validate transaction data before prediction"""
        errors = []
        
        # Check if all required features are present
        for feature in required_features:
            if feature not in transaction_data:
                errors.append(f"Missing required feature: {feature}")
        
        # Check for invalid values
        for feature, value in transaction_data.items():
            if pd.isna(value):
                errors.append(f"Invalid value for {feature}: NaN")
            elif not isinstance(value, (int, float)):
                errors.append(f"Invalid data type for {feature}: {type(value)}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    @staticmethod
    def export_results_to_csv(results_df, filename=None):
        """Export results to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fraud_detection_results_{timestamp}.csv"
        
        try:
            results_df.to_csv(filename, index=False)
            return True, filename
        except Exception as e:
            st.error(f"Error exporting to CSV: {str(e)}")
            return False, None
    
    @staticmethod
    def create_performance_summary(results):
        """Create a comprehensive performance summary"""
        summary = {}
        
        for model_name, metrics in results.items():
            summary[model_name] = {
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
                'AUC-ROC': f"{metrics['auc_roc']:.4f}",
                'Performance Grade': Utils._calculate_performance_grade(metrics)
            }
        
        return summary
    
    @staticmethod
    def _calculate_performance_grade(metrics):
        """Calculate performance grade based on metrics"""
        avg_score = (metrics['accuracy'] + metrics['precision'] + 
                    metrics['recall'] + metrics['f1'] + metrics['auc_roc']) / 5
        
        if avg_score >= 0.95:
            return "A+"
        elif avg_score >= 0.90:
            return "A"
        elif avg_score >= 0.85:
            return "B+"
        elif avg_score >= 0.80:
            return "B"
        elif avg_score >= 0.75:
            return "C+"
        elif avg_score >= 0.70:
            return "C"
        else:
            return "D"
    
    @staticmethod
    def get_model_recommendations(results):
        """Get recommendations for model selection"""
        recommendations = []
        
        # Find best model by AUC-ROC
        best_auc_model = max(results.keys(), key=lambda k: results[k]['auc_roc'])
        recommendations.append(f"Best overall performance: {best_auc_model} (AUC-ROC: {results[best_auc_model]['auc_roc']:.4f})")
        
        # Find best precision model
        best_precision_model = max(results.keys(), key=lambda k: results[k]['precision'])
        if results[best_precision_model]['precision'] > 0.9:
            recommendations.append(f"Best for minimizing false positives: {best_precision_model} (Precision: {results[best_precision_model]['precision']:.4f})")
        
        # Find best recall model
        best_recall_model = max(results.keys(), key=lambda k: results[k]['recall'])
        if results[best_recall_model]['recall'] > 0.9:
            recommendations.append(f"Best for catching fraud: {best_recall_model} (Recall: {results[best_recall_model]['recall']:.4f})")
        
        return recommendations
    
    @staticmethod
    def simulate_real_time_monitoring():
        """Simulate real-time monitoring dashboard data"""
        current_time = datetime.now()
        
        # Generate hourly transaction data for the last 24 hours
        hours = []
        transaction_counts = []
        fraud_counts = []
        
        for i in range(24):
            hour_time = current_time - timedelta(hours=23-i)
            hours.append(hour_time.strftime("%H:00"))
            
            # Simulate transaction volume (higher during business hours)
            if 8 <= hour_time.hour <= 20:
                base_transactions = np.random.randint(800, 1200)
            else:
                base_transactions = np.random.randint(200, 400)
            
            transaction_counts.append(base_transactions)
            
            # Simulate fraud (typically 1-3% of transactions)
            fraud_count = int(base_transactions * np.random.uniform(0.01, 0.03))
            fraud_counts.append(fraud_count)
        
        return {
            'hours': hours,
            'transactions': transaction_counts,
            'fraud_detected': fraud_counts,
            'total_transactions': sum(transaction_counts),
            'total_fraud': sum(fraud_counts)
        }
