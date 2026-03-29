import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from footer import render_footer
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ensemble_methods import EnsembleModel

st.set_page_config(page_title="Ensemble Methods", page_icon="🤝", layout="wide")

st.title("🤝 Ensemble Methods for Improved Accuracy")
st.markdown("Combine multiple models to achieve better fraud detection performance")

# Check if models are trained
if 'models_trained' not in st.session_state or not st.session_state.models_trained:
    st.warning("⚠️ Please train individual models first from the Model Training page.")
    st.stop()

if 'model_trainer' not in st.session_state or st.session_state.model_trainer is None:
    st.error("Model trainer not found. Please retrain models.")
    st.stop()

# Initialize ensemble model in session state
if 'ensemble_model' not in st.session_state:
    st.session_state.ensemble_model = EnsembleModel()

model_trainer = st.session_state.model_trainer
ensemble_model = st.session_state.ensemble_model
data_processor = st.session_state.data_processor

# Get available models (excluding neural networks for voting/stacking)
# Note: VotingClassifier/StackingClassifier require names without spaces — use underscored keys
available_models = {}
nn_excluded = False
for name, model in model_trainer.models.items():
    if name in ('Neural Network', 'Deep Neural Network', 'deep_nn'):
        nn_excluded = True
    else:
        # Sanitize name: replace spaces/special chars with underscores
        safe_name = name.replace(' ', '_').lower()
        available_models[safe_name] = model

# All models (including NN) for weighted ensemble
all_models = {}
for name, model in model_trainer.models.items():
    safe_name = name.replace(' ', '_').lower()
    all_models[safe_name] = model

# Show info about excluded models
if nn_excluded:
    st.info("ℹ️ **Note:** Neural Network models are excluded from Voting and Stacking ensembles due to API compatibility. They can be included in Weighted Average ensembles.")

st.markdown("---")

# Ensemble Method Selection
st.header("📊 Select Ensemble Method")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Voting Ensemble")
    st.write("Combines predictions through voting (majority or weighted)")
    voting_type = st.selectbox("Voting Type", ["soft", "hard"], 
                               help="Soft: Uses predicted probabilities, Hard: Uses majority vote")
    train_voting = st.button("Train Voting Ensemble", type="primary")

with col2:
    st.subheader("Stacking Ensemble")
    st.write("Uses a meta-learner to combine model predictions")
    meta_learner_type = st.selectbox("Meta-Learner", ["Logistic Regression", "Random Forest"],
                                     help="Model to combine base predictions")
    train_stacking = st.button("Train Stacking Ensemble", type="primary")

with col3:
    st.subheader("Weighted Average")
    st.write("Weighted average of model probabilities")
    use_optimized_weights = st.checkbox("Optimize Weights", value=True,
                                        help="Automatically find best weights")
    train_weighted = st.button("Train Weighted Ensemble", type="primary")

st.markdown("---")

# Training section
if train_voting:
    with st.spinner("Training Voting Ensemble..."):
        if len(available_models) < 2:
            st.error("Need at least 2 models for ensemble. Train more models first.")
        elif model_trainer.X_train is None:
            st.error("Training data not found. Please retrain models first.")
        else:
            # Use stored training data from model trainer
            X_train = model_trainer.X_train
            X_test = model_trainer.X_test
            y_train = model_trainer.y_train
            y_test = model_trainer.y_test
            
            # Train voting ensemble
            metrics = ensemble_model.train_voting_ensemble(
                X_train, X_test, y_train, y_test, 
                available_models, 
                voting_type=voting_type
            )
            
            st.session_state.ensemble_trained = True
            st.success("✅ Voting Ensemble trained successfully!")
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']:.4f}")
            with col5:
                st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")

if train_stacking:
    with st.spinner("Training Stacking Ensemble..."):
        if len(available_models) < 2:
            st.error("Need at least 2 models for ensemble. Train more models first.")
        elif model_trainer.X_train is None:
            st.error("Training data not found. Please retrain models first.")
        else:
            # Use stored training data from model trainer
            X_train = model_trainer.X_train
            X_test = model_trainer.X_test
            y_train = model_trainer.y_train
            y_test = model_trainer.y_test
            
            # Create meta-learner
            if meta_learner_type == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            else:
                from sklearn.ensemble import RandomForestClassifier
                meta_learner = RandomForestClassifier(n_estimators=50, random_state=42)
            
            # Train stacking ensemble
            metrics = ensemble_model.train_stacking_ensemble(
                X_train, X_test, y_train, y_test, 
                available_models, 
                meta_learner=meta_learner
            )
            
            st.session_state.ensemble_trained = True
            st.success("✅ Stacking Ensemble trained successfully!")
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']:.4f}")
            with col5:
                st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")

if train_weighted:
    with st.spinner("Training Weighted Average Ensemble..."):
        if len(all_models) < 2:
            st.error("Need at least 2 models for ensemble. Train more models first.")
        elif model_trainer.X_train is None:
            st.error("Training data not found. Please retrain models first.")
        else:
            # Use stored training data from model trainer
            X_train = model_trainer.X_train
            X_test = model_trainer.X_test
            y_train = model_trainer.y_train
            y_test = model_trainer.y_test
            
            weights = None
            if use_optimized_weights:
                # Split train set for validation (use train data for weight optimization)
                from sklearn.model_selection import train_test_split
                X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                
                weights, best_score = ensemble_model.optimize_weights(X_val, y_val, all_models)
                st.info(f"Optimized weights found (Validation AUC: {best_score:.4f})")
                
                # Show weights
                weights_df = pd.DataFrame(list(weights.items()), columns=['Model', 'Weight'])
                st.dataframe(weights_df)
                
                # Evaluate with optimized weights on test set
                metrics = ensemble_model.train_weighted_ensemble(
                    X_test, y_test, all_models, weights=weights
                )
            else:
                metrics = ensemble_model.train_weighted_ensemble(
                    X_test, y_test, all_models
                )
            
            st.session_state.ensemble_trained = True
            st.success("✅ Weighted Average Ensemble trained successfully!")
            
            # Display metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']:.4f}")
            with col5:
                st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")

# Ensemble Results Comparison
if ensemble_model.ensemble_results:
    st.markdown("---")
    st.header("📈 Ensemble Performance Comparison")
    
    # Get comparison dataframe
    comparison_df = ensemble_model.get_ensemble_comparison()
    
    # Add individual model results for comparison
    if 'training_results' in st.session_state:
        individual_results = st.session_state.training_results
        for model, metrics in individual_results.items():
            comparison_df.loc[model] = {
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc']
            }
    
    # Display comparison table
    st.subheader("Performance Metrics Comparison")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))
    
    # Create visualization
    st.subheader("Visual Comparison")
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig = go.Figure()
    
    for metric in metrics_to_plot:
        fig.add_trace(go.Scatter(
            x=comparison_df.index,
            y=comparison_df[metric],
            mode='lines+markers',
            name=metric,
            line=dict(width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title='Model and Ensemble Performance Comparison',
        xaxis_title='Model / Ensemble',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve Comparison
    st.subheader("ROC Curve Comparison")
    
    roc_fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    color_idx = 0
    
    # Add individual model ROC curves
    if 'training_results' in st.session_state:
        for model_name, metrics in st.session_state.training_results.items():
            if 'roc_data' in metrics:
                fpr, tpr, _ = metrics['roc_data']
                roc_fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name}',
                    line=dict(width=2, color=colors[color_idx % len(colors)])
                ))
                color_idx += 1
    
    # Add ensemble ROC curves
    for ensemble_name, metrics in ensemble_model.ensemble_results.items():
        if 'roc_data' in metrics:
            fpr, tpr, _ = metrics['roc_data']
            roc_fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{ensemble_name}',
                line=dict(width=3, color=colors[color_idx % len(colors)], dash='dash')
            ))
            color_idx += 1
    
    # Add diagonal line
    roc_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='black', width=1)
    ))
    
    roc_fig.update_layout(
        title='ROC Curves: Individual Models vs Ensembles',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )
    
    st.plotly_chart(roc_fig, use_container_width=True)
    
    # Best Model Recommendation
    st.markdown("---")
    st.header("🏆 Best Model Recommendation")
    
    best_model = comparison_df['AUC-ROC'].idxmax()
    best_auc = comparison_df.loc[best_model, 'AUC-ROC']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.success(f"**Best Model:** {best_model}")
        st.metric("AUC-ROC Score", f"{best_auc:.4f}")
    
    with col2:
        st.write("**Performance Analysis:**")
        model_metrics = comparison_df.loc[best_model]
        
        if model_metrics['Precision'] > 0.9:
            st.write("✅ Excellent precision - Minimizes false fraud alerts")
        if model_metrics['Recall'] > 0.9:
            st.write("✅ Excellent recall - Catches most fraud cases")
        if model_metrics['F1-Score'] > 0.9:
            st.write("✅ Well-balanced performance across metrics")
        
        if best_model in ensemble_model.ensemble_results:
            st.write("🤝 **This is an ensemble method, combining the strengths of multiple models!**")

# Model Contribution Analysis
if ensemble_model.ensemble_results and all_models and model_trainer.X_test is not None:
    st.markdown("---")
    st.header("🔍 Individual Model Contributions")
    
    X_test = model_trainer.X_test
    y_test = model_trainer.y_test
    
    contributions = ensemble_model.get_model_contributions(X_test, y_test, all_models)
    
    contrib_df = pd.DataFrame(list(contributions.items()), columns=['Model', 'AUC-ROC'])
    contrib_df = contrib_df.sort_values('AUC-ROC', ascending=False)
    
    fig = go.Figure(go.Bar(
        x=contrib_df['AUC-ROC'],
        y=contrib_df['Model'],
        orientation='h',
        marker_color='#45B7D1'
    ))
    
    fig.update_layout(
        title='Individual Model Performance Contribution',
        xaxis_title='AUC-ROC Score',
        yaxis_title='Model',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Information Section
st.markdown("---")
st.header("ℹ️ About Ensemble Methods")

with st.expander("Voting Ensemble"):
    st.markdown("""
    **Voting Ensemble** combines predictions from multiple models through voting:
    
    - **Hard Voting**: Each model votes for a class, final prediction is the majority vote
    - **Soft Voting**: Averages predicted probabilities from all models (recommended)
    
    **Advantages:**
    - Simple and effective
    - Reduces variance and improves stability
    - Works well when base models have similar performance
    """)

with st.expander("Stacking Ensemble"):
    st.markdown("""
    **Stacking Ensemble** uses a meta-learner to combine predictions:
    
    - Base models make predictions on the training data
    - Meta-learner learns how to best combine these predictions
    - Can capture complex relationships between model outputs
    
    **Advantages:**
    - Often achieves best performance
    - Can learn optimal way to combine diverse models
    - Flexible meta-learner selection
    """)

with st.expander("Weighted Average"):
    st.markdown("""
    **Weighted Average** assigns different weights to model predictions:
    
    - Each model's prediction is multiplied by its weight
    - Weights sum to 1.0
    - Can be optimized based on validation performance
    
    **Advantages:**
    - Allows emphasizing better-performing models
    - Weight optimization can improve results
    - Interpretable and transparent
    """)

with st.expander("When to Use Ensemble Methods"):
    st.markdown("""
    Ensemble methods work best when:
    
    1. **Diverse Models**: You have models with different strengths/weaknesses
    2. **Similar Performance**: Individual models have comparable accuracy
    3. **Complex Patterns**: The fraud patterns are complex and multifaceted
    4. **High Stakes**: Maximum accuracy is critical (financial applications)
    
    **General Guidelines:**
    - Start with Voting Ensemble for simplicity
    - Use Stacking for maximum performance
    - Try Weighted Average when you know model strengths
    """)

render_footer()
