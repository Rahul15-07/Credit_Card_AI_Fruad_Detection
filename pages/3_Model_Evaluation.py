import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from footer import render_footer
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")

st.title("📊 Model Performance Evaluation")
st.markdown("Comprehensive evaluation of trained models using multiple metrics and visualizations.")

# Check if models are trained
if not st.session_state.get('models_trained', False):
    st.warning("⚠️ Please train models first.")
    st.stop()

trained_models = st.session_state.trained_models
X_test = st.session_state.X_test
y_test = st.session_state.y_test

st.markdown("---")

# Generate predictions for all models
@st.cache_data
def generate_predictions():
    predictions = {}
    prediction_probas = {}
    
    for model_name, model in trained_models.items():
        if model_name == 'deep_nn':
            # TensorFlow model predictions
            pred_proba = model.predict(X_test)
            pred = (pred_proba > 0.5).astype(int).flatten()
            predictions[model_name] = pred
            prediction_probas[model_name] = pred_proba.flatten()
        else:
            # Scikit-learn model predictions
            pred = model.predict(X_test)
            pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of fraud
            predictions[model_name] = pred
            prediction_probas[model_name] = pred_proba
    
    return predictions, prediction_probas

predictions, prediction_probas = generate_predictions()

# Calculate metrics for all models
@st.cache_data
def calculate_metrics():
    metrics_df = []
    
    for model_name in trained_models.keys():
        y_pred = predictions[model_name]
        y_proba = prediction_probas[model_name]
        
        metrics = {
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_proba)
        }
        metrics_df.append(metrics)
    
    return pd.DataFrame(metrics_df)

metrics_df = calculate_metrics()

# Performance Summary
st.subheader("🎯 Performance Summary")

# Metrics table
st.dataframe(
    metrics_df.round(4).style.format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-Score': '{:.4f}',
        'AUC-ROC': '{:.4f}'
    }).highlight_max(axis=0, props='background-color: lightgreen'),
    use_container_width=True,
    hide_index=True
)

# Best performing model
best_model_idx = metrics_df['F1-Score'].idxmax()
best_model = metrics_df.loc[best_model_idx, 'Model']
best_f1 = metrics_df.loc[best_model_idx, 'F1-Score']

st.success(f"🏆 Best performing model: **{best_model}** (F1-Score: {best_f1:.4f})")

st.markdown("---")

# Metric Comparisons
st.subheader("📈 Metric Comparisons")

# Create comparison charts
col1, col2 = st.columns(2)

with col1:
    # Accuracy and AUC-ROC comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=metrics_df['Model'],
        y=metrics_df['Accuracy'],
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        name='AUC-ROC',
        x=metrics_df['Model'],
        y=metrics_df['AUC-ROC'],
        marker_color='orange'
    ))
    fig.update_layout(
        title='Accuracy vs AUC-ROC Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Precision, Recall, F1-Score comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Precision',
        x=metrics_df['Model'],
        y=metrics_df['Precision'],
        marker_color='red'
    ))
    fig.add_trace(go.Bar(
        name='Recall',
        x=metrics_df['Model'],
        y=metrics_df['Recall'],
        marker_color='green'
    ))
    fig.add_trace(go.Bar(
        name='F1-Score',
        x=metrics_df['Model'],
        y=metrics_df['F1-Score'],
        marker_color='purple'
    ))
    fig.update_layout(
        title='Precision, Recall, F1-Score Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

# Radar chart for overall comparison
st.subheader("🎯 Overall Performance Radar Chart")

selected_models = st.multiselect(
    "Select models for radar chart comparison",
    metrics_df['Model'].tolist(),
    default=metrics_df['Model'].tolist()[:3]
)

if selected_models:
    fig = go.Figure()
    
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    for model in selected_models:
        model_data = metrics_df[metrics_df['Model'] == model].iloc[0]
        values = [model_data[col] for col in metrics_cols]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_cols,
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ROC Curves
st.subheader("📊 ROC Curves Analysis")

fig = go.Figure()

# Add ROC curve for each model
for model_name in trained_models.keys():
    y_proba = prediction_probas[model_name]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        name=f"{model_name.replace('_', ' ').title()} (AUC: {auc_score:.4f})",
        mode='lines'
    ))

# Add diagonal line (random classifier)
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    line=dict(dash='dash', color='red'),
    name='Random Classifier (AUC: 0.5000)'
))

fig.update_layout(
    title='ROC Curves Comparison',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    width=800,
    height=600
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Confusion Matrices
st.subheader("🔍 Confusion Matrix Analysis")

selected_model_cm = st.selectbox(
    "Select model for confusion matrix",
    list(trained_models.keys()),
    format_func=lambda x: x.replace('_', ' ').title()
)

if selected_model_cm:
    y_pred = predictions[selected_model_cm]
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix heatmap
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Predicted Legitimate', 'Predicted Fraudulent'],
            y=['Actual Legitimate', 'Actual Fraudulent'],
            colorscale='Blues',
            showscale=True
        )
        fig.update_layout(
            title=f'Confusion Matrix - {selected_model_cm.replace("_", " ").title()}',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion matrix metrics breakdown
        tn, fp, fn, tp = cm.ravel()
        
        st.metric("True Negatives (Correct Legitimate)", tn)
        st.metric("False Positives (Incorrect Fraud Alert)", fp)
        st.metric("False Negatives (Missed Fraud)", fn)
        st.metric("True Positives (Correct Fraud Detection)", tp)
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        st.markdown("---")
        st.metric("Specificity (True Negative Rate)", f"{specificity:.4f}")
        st.metric("Sensitivity (True Positive Rate)", f"{sensitivity:.4f}")

st.markdown("---")

# Classification Reports
st.subheader("📋 Detailed Classification Reports")

selected_model_report = st.selectbox(
    "Select model for detailed classification report",
    list(trained_models.keys()),
    format_func=lambda x: x.replace('_', ' ').title()
)

if selected_model_report:
    y_pred = predictions[selected_model_report]
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Convert to DataFrame for better display
    report_df = pd.DataFrame(report).transpose()
    
    # Format the DataFrame
    formatted_report = report_df.round(4)
    
    st.dataframe(
        formatted_report.style.format({
            'precision': '{:.4f}',
            'recall': '{:.4f}',
            'f1-score': '{:.4f}',
            'support': '{:.0f}'
        }),
        use_container_width=True
    )

# Threshold Analysis
st.subheader("⚖️ Classification Threshold Analysis")

threshold_model = st.selectbox(
    "Select model for threshold analysis",
    list(trained_models.keys()),
    format_func=lambda x: x.replace('_', ' ').title()
)

if threshold_model:
    y_proba = prediction_probas[threshold_model]
    
    # Calculate metrics for different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    threshold_metrics = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        
        threshold_metrics.append({
            'Threshold': threshold,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    threshold_df = pd.DataFrame(threshold_metrics)
    
    # Plot threshold analysis
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=threshold_df['Threshold'],
        y=threshold_df['Precision'],
        name='Precision',
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=threshold_df['Threshold'],
        y=threshold_df['Recall'],
        name='Recall',
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=threshold_df['Threshold'],
        y=threshold_df['F1-Score'],
        name='F1-Score',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title=f'Threshold Analysis - {threshold_model.replace("_", " ").title()}',
        xaxis_title='Classification Threshold',
        yaxis_title='Score',
        hovermode='x'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Optimal threshold
    optimal_idx = threshold_df['F1-Score'].idxmax()
    optimal_threshold = threshold_df.loc[optimal_idx, 'Threshold']
    optimal_f1 = threshold_df.loc[optimal_idx, 'F1-Score']
    
    st.info(f"🎯 Optimal threshold for {threshold_model.replace('_', ' ').title()}: {optimal_threshold:.2f} (F1-Score: {optimal_f1:.4f})")

# Model insights and recommendations
st.markdown("---")
st.subheader("💡 Model Insights and Recommendations")

# Performance insights
with st.expander("Performance Analysis"):
    st.markdown(f"""
    **Best Overall Model**: {best_model}
    
    **Key Insights**:
    - The model with the highest F1-Score provides the best balance between precision and recall
    - Consider the business cost of false positives vs false negatives when selecting the final model
    - AUC-ROC score indicates the model's ability to distinguish between classes
    
    **Recommendations**:
    - Use the confusion matrix to understand the types of errors each model makes
    - Consider ensemble methods if individual models show complementary strengths
    - Monitor model performance over time and retrain as needed
    """)

# Business impact analysis
with st.expander("Business Impact Analysis"):
    # Calculate business metrics
    for model_name in trained_models.keys():
        y_pred = predictions[model_name]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Assuming average fraud amount and processing costs
        avg_fraud_amount = 100  # This would be calculated from actual data
        processing_cost = 5     # Cost of investigating each alert
        
        cost_saved = tp * avg_fraud_amount  # Fraud prevented
        investigation_cost = (fp + tp) * processing_cost  # Cost of all alerts
        missed_fraud_cost = fn * avg_fraud_amount  # Cost of missed fraud
        
        net_benefit = cost_saved - investigation_cost - missed_fraud_cost
        
        st.markdown(f"""
        **{model_name.replace('_', ' ').title()}**:
        - Fraud prevented: ${cost_saved:,.2f}
        - Investigation costs: ${investigation_cost:,.2f}
        - Missed fraud cost: ${missed_fraud_cost:,.2f}
        - Net benefit: ${net_benefit:,.2f}
        """)

st.info("🎯 Model evaluation complete! Navigate to 'Real-Time Prediction' to test individual transactions.")

render_footer()
