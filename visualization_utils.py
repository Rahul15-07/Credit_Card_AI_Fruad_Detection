import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class VisualizationUtils:
    def __init__(self):
        self.color_scheme = {
            'fraud': '#FF6B6B',
            'legitimate': '#4ECDC4',
            'primary': '#45B7D1',
            'secondary': '#96CEB4',
            'accent': '#FFEAA7'
        }
    
    def plot_class_distribution(self, data):
        """Plot the distribution of fraud vs legitimate transactions"""
        fraud_counts = data['is_fraud'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Legitimate', 'Fraudulent'],
                y=[fraud_counts[0], fraud_counts[1]],
                marker_color=[self.color_scheme['legitimate'], self.color_scheme['fraud']],
                text=[f'{fraud_counts[0]:,}', f'{fraud_counts[1]:,}'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Transaction Class Distribution',
            xaxis_title='Transaction Type',
            yaxis_title='Number of Transactions',
            showlegend=False
        )
        
        return fig
    
    def plot_correlation_heatmap(self, data):
        """Plot correlation heatmap of features"""
        # Calculate correlation matrix (excluding non-numeric columns)
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.around(corr_matrix.values, decimals=2),
            texttemplate='%{text}',
            textfont={'size': 8},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            width=800,
            height=600
        )
        
        return fig
    
    def plot_model_comparison(self, results):
        """Plot comparison of model performance metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        models = list(results.keys())
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results[model][metric] for model in models]
            fig.add_trace(go.Scatter(
                x=models,
                y=values,
                mode='lines+markers',
                name=metric.upper(),
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            legend=dict(x=0, y=1),
            hovermode='x unified'
        )
        
        return fig
    
    def plot_roc_curves(self, results):
        """Plot ROC curves for all models"""
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (model_name, metrics) in enumerate(results.items()):
            if 'roc_data' in metrics:
                fpr, tpr, _ = metrics['roc_data']
                auc_score = metrics['auc_roc']
                
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc_score:.3f})',
                    line=dict(width=3, color=colors[i % len(colors)])
                ))
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='black', width=2)
        ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance for tree-based models"""
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color=self.color_scheme['primary']
        ))
        
        fig.update_layout(
            title='Feature Importance (Random Forest)',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix, model_name):
        """Plot confusion matrix"""
        labels = ['Legitimate', 'Fraudulent']
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=confusion_matrix,
            texttemplate='%{text}',
            textfont={'size': 16}
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=400,
            height=400
        )
        
        return fig
    
    def plot_fraud_distribution_by_feature(self, data, feature):
        """Plot fraud distribution by a specific feature"""
        fig = go.Figure()
        
        # Legitimate transactions
        legitimate_data = data[data['is_fraud'] == 0][feature]
        fig.add_trace(go.Histogram(
            x=legitimate_data,
            name='Legitimate',
            opacity=0.7,
            marker_color=self.color_scheme['legitimate'],
            nbinsx=30
        ))
        
        # Fraudulent transactions
        fraud_data = data[data['is_fraud'] == 1][feature]
        fig.add_trace(go.Histogram(
            x=fraud_data,
            name='Fraudulent',
            opacity=0.7,
            marker_color=self.color_scheme['fraud'],
            nbinsx=30
        ))
        
        fig.update_layout(
            title=f'Distribution of {feature} by Transaction Type',
            xaxis_title=feature,
            yaxis_title='Frequency',
            barmode='overlay'
        )
        
        return fig
    
    def plot_prediction_probabilities(self, probabilities, predictions):
        """Plot distribution of prediction probabilities"""
        fig = go.Figure()
        
        # Separate probabilities by actual prediction
        legitimate_probs = [p for p, pred in zip(probabilities, predictions) if pred == 0]
        fraud_probs = [p for p, pred in zip(probabilities, predictions) if pred == 1]
        
        fig.add_trace(go.Histogram(
            x=legitimate_probs,
            name='Predicted Legitimate',
            opacity=0.7,
            marker_color=self.color_scheme['legitimate'],
            nbinsx=20
        ))
        
        fig.add_trace(go.Histogram(
            x=fraud_probs,
            name='Predicted Fraudulent',
            opacity=0.7,
            marker_color=self.color_scheme['fraud'],
            nbinsx=20
        ))
        
        fig.update_layout(
            title='Distribution of Prediction Probabilities',
            xaxis_title='Fraud Probability',
            yaxis_title='Count',
            barmode='overlay'
        )
        
        return fig
    
    def create_metrics_radar_chart(self, results):
        """Create radar chart comparing model metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        models = list(results.keys())
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, model in enumerate(models):
            values = [results[model][metric] for metric in metrics]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        return fig
    
    def plot_training_history(self, history):
        """Plot training history for neural network"""
        if not history:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Loss', 'Precision', 'Recall')
        )
        
        epochs = range(1, len(history['accuracy']) + 1)
        
        # Accuracy
        fig.add_trace(go.Scatter(x=list(epochs), y=history['accuracy'], name='Train Accuracy'), row=1, col=1)
        if 'val_accuracy' in history:
            fig.add_trace(go.Scatter(x=list(epochs), y=history['val_accuracy'], name='Val Accuracy'), row=1, col=1)
        
        # Loss
        fig.add_trace(go.Scatter(x=list(epochs), y=history['loss'], name='Train Loss'), row=1, col=2)
        if 'val_loss' in history:
            fig.add_trace(go.Scatter(x=list(epochs), y=history['val_loss'], name='Val Loss'), row=1, col=2)
        
        # Precision
        fig.add_trace(go.Scatter(x=list(epochs), y=history['precision'], name='Train Precision'), row=2, col=1)
        if 'val_precision' in history:
            fig.add_trace(go.Scatter(x=list(epochs), y=history['val_precision'], name='Val Precision'), row=2, col=1)
        
        # Recall
        fig.add_trace(go.Scatter(x=list(epochs), y=history['recall'], name='Train Recall'), row=2, col=2)
        if 'val_recall' in history:
            fig.add_trace(go.Scatter(x=list(epochs), y=history['val_recall'], name='Val Recall'), row=2, col=2)
        
        fig.update_layout(height=600, title="Neural Network Training History")
        
        return fig
