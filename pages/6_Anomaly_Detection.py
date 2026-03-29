import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from footer import render_footer
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
sys.path.append('..')
from anomaly_detector import AnomalyDetector

st.set_page_config(page_title="Anomaly Detection", page_icon="🔍", layout="wide")

st.title("🔍 Anomaly Detection for Fraud Detection")
st.markdown("Use unsupervised learning to identify unusual transaction patterns")

# Check if data is loaded
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.warning("⚠️ Please generate a dataset first from the Home page.")
    st.stop()

# Initialize anomaly detector in session state
if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector()

anomaly_detector = st.session_state.anomaly_detector
data_processor = st.session_state.data_processor

st.markdown("---")

# Method Selection
st.header("📊 Select Anomaly Detection Method")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌲 Isolation Forest")
    st.write("""
    Tree-based algorithm that isolates anomalies by randomly selecting features and split values.
    
    **Advantages:**
    - Fast and efficient
    - Works well with high-dimensional data
    - No assumptions about data distribution
    """)
    
    with st.expander("Configure Isolation Forest"):
        contamination = st.slider(
            "Expected Fraud Rate (Contamination)",
            min_value=0.01, max_value=0.5, value=0.1, step=0.01,
            help="Expected proportion of fraud in the dataset"
        )
        n_estimators = st.number_input(
            "Number of Trees",
            min_value=50, max_value=500, value=100, step=50
        )
    
    train_if = st.button("Train Isolation Forest", type="primary")

with col2:
    st.subheader("🧠 Autoencoder")
    st.write("""
    Neural network that learns to reconstruct normal patterns. High reconstruction error indicates anomaly.
    
    **Advantages:**
    - Captures complex patterns
    - Learns feature interactions
    - Provides reconstruction insights
    """)
    
    with st.expander("Configure Autoencoder"):
        encoding_dim = st.number_input(
            "Encoding Dimension",
            min_value=5, max_value=50, value=10, step=5,
            help="Size of the compressed representation"
        )
        ae_epochs = st.number_input(
            "Training Epochs",
            min_value=10, max_value=100, value=50, step=10
        )
        ae_batch_size = st.number_input(
            "Batch Size",
            min_value=16, max_value=128, value=32, step=16
        )
    
    train_ae = st.button("Train Autoencoder", type="primary")

st.markdown("---")

# Store anomaly detection split in session state if not exists
if 'anomaly_X_train' not in st.session_state:
    X_train, X_test, y_train, y_test = data_processor.preprocess_data(
        test_size=0.2, use_smote=False, random_state=42  # No SMOTE for anomaly detection
    )
    st.session_state.anomaly_X_train = X_train
    st.session_state.anomaly_X_test = X_test
    st.session_state.anomaly_y_train = y_train
    st.session_state.anomaly_y_test = y_test
    st.info("ℹ️ **Note:** Both anomaly detection models will be evaluated on the same train/test split for fair comparison.")

# Training section
if train_if:
    with st.spinner("Training Isolation Forest..."):
        # Use cached anomaly detection data
        X_train = st.session_state.anomaly_X_train
        X_test = st.session_state.anomaly_X_test
        y_train = st.session_state.anomaly_y_train
        y_test = st.session_state.anomaly_y_test
        
        # Train Isolation Forest
        metrics = anomaly_detector.train_isolation_forest(
            X_train, X_test, y_train, y_test,
            contamination=contamination,
            n_estimators=n_estimators
        )
        
        st.session_state.anomaly_trained = True
        st.success("✅ Isolation Forest trained successfully!")
        
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

if train_ae:
    with st.spinner("Training Autoencoder... This may take a few minutes."):
        # Use cached anomaly detection data
        X_train = st.session_state.anomaly_X_train
        X_test = st.session_state.anomaly_X_test
        y_train = st.session_state.anomaly_y_train
        y_test = st.session_state.anomaly_y_test
        
        # Train Autoencoder
        metrics = anomaly_detector.train_autoencoder(
            X_train, X_test, y_train, y_test,
            encoding_dim=encoding_dim,
            epochs=ae_epochs,
            batch_size=ae_batch_size
        )
        
        st.session_state.anomaly_trained = True
        st.success("✅ Autoencoder trained successfully!")
        
        # Display metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
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
        with col6:
            st.metric("Threshold", f"{metrics['threshold']:.6f}")

# Results Comparison
if anomaly_detector.results:
    st.markdown("---")
    st.header("📈 Anomaly Detection Performance")
    
    # Comparison table
    comparison_df = anomaly_detector.compare_methods()
    
    st.subheader("Performance Metrics Comparison")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics comparison chart
        metrics_fig = go.Figure()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        
        for metric in metrics_to_plot:
            metrics_fig.add_trace(go.Scatter(
                x=comparison_df.index,
                y=comparison_df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        metrics_fig.update_layout(
            title='Anomaly Detection Methods Comparison',
            xaxis_title='Method',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        st.plotly_chart(metrics_fig, use_container_width=True)
    
    with col2:
        # ROC curves
        roc_fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e']
        
        for idx, (method, metrics) in enumerate(anomaly_detector.results.items()):
            if 'roc_data' in metrics:
                fpr, tpr, _ = metrics['roc_data']
                auc_score = metrics['auc_roc']
                
                roc_fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{method} (AUC = {auc_score:.3f})',
                    line=dict(width=3, color=colors[idx])
                ))
        
        # Add diagonal
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='black')
        ))
        
        roc_fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        
        st.plotly_chart(roc_fig, use_container_width=True)
    
    # Score distributions
    st.markdown("---")
    st.subheader("📊 Score Distributions")
    
    # Use cached anomaly detection data
    X_test = st.session_state.anomaly_X_test
    y_test = st.session_state.anomaly_y_test
    
    if 'Isolation Forest' in anomaly_detector.results:
        st.write("**Isolation Forest Anomaly Scores**")
        
        score_dist = anomaly_detector.get_anomaly_scores_distribution(X_test, y_test)
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=score_dist['normal_scores'],
            name='Normal Transactions',
            opacity=0.7,
            marker_color='green',
            nbinsx=50
        ))
        
        fig.add_trace(go.Histogram(
            x=score_dist['fraud_scores'],
            name='Fraud Transactions',
            opacity=0.7,
            marker_color='red',
            nbinsx=50
        ))
        
        fig.update_layout(
            title='Anomaly Score Distribution (Lower = More Anomalous)',
            xaxis_title='Anomaly Score',
            yaxis_title='Frequency',
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if 'Autoencoder' in anomaly_detector.results:
        st.write("**Autoencoder Reconstruction Errors**")
        
        error_dist = anomaly_detector.get_reconstruction_error_distribution(X_test, y_test)
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=error_dist['normal_errors'],
            name='Normal Transactions',
            opacity=0.7,
            marker_color='green',
            nbinsx=50
        ))
        
        fig.add_trace(go.Histogram(
            x=error_dist['fraud_errors'],
            name='Fraud Transactions',
            opacity=0.7,
            marker_color='red',
            nbinsx=50
        ))
        
        # Add threshold line
        fig.add_vline(
            x=error_dist['threshold'],
            line_dash="dash",
            line_color="black",
            annotation_text=f"Threshold: {error_dist['threshold']:.6f}"
        )
        
        fig.update_layout(
            title='Reconstruction Error Distribution (Higher = More Anomalous)',
            xaxis_title='Reconstruction Error (MSE)',
            yaxis_title='Frequency',
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Training history for autoencoder
    if 'Autoencoder' in anomaly_detector.results and 'history' in anomaly_detector.results['Autoencoder']:
        st.markdown("---")
        st.subheader("📉 Autoencoder Training History")
        
        history = anomaly_detector.results['Autoencoder']['history']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Loss', 'Validation Loss')
        )
        
        epochs = range(1, len(history['loss']) + 1)
        
        fig.add_trace(
            go.Scatter(x=list(epochs), y=history['loss'], name='Train Loss'),
            row=1, col=1
        )
        
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(x=list(epochs), y=history['val_loss'], name='Val Loss'),
                row=1, col=2
            )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model recommendation
    st.markdown("---")
    st.header("🏆 Best Anomaly Detection Method")
    
    best_method = comparison_df['AUC-ROC'].idxmax()
    best_auc = comparison_df.loc[best_method, 'AUC-ROC']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.success(f"**Best Method:** {best_method}")
        st.metric("AUC-ROC Score", f"{best_auc:.4f}")
    
    with col2:
        st.write("**Performance Analysis:**")
        
        if best_method == 'Isolation Forest':
            st.write("""
            - ✅ Fast and efficient for real-time detection
            - ✅ Works well with high-dimensional data
            - ✅ No assumptions about data distribution
            - ⚠️ Less interpretable than reconstruction-based methods
            """)
        else:
            st.write("""
            - ✅ Captures complex, non-linear patterns
            - ✅ Provides reconstruction insights
            - ✅ Can visualize what makes a transaction anomalous
            - ⚠️ Requires more computational resources
            - ⚠️ May need careful threshold tuning
            """)

# Information Section
st.markdown("---")
st.header("ℹ️ About Anomaly Detection")

with st.expander("What is Anomaly Detection?"):
    st.markdown("""
    Anomaly detection identifies data points that differ significantly from the majority of the data.
    In fraud detection, anomalies often correspond to fraudulent transactions.
    
    **Key Advantages:**
    - **Unsupervised**: Doesn't require labeled fraud data
    - **Adaptable**: Can detect new, unknown fraud patterns
    - **Real-time**: Fast enough for transaction processing
    """)

with st.expander("Isolation Forest Explained"):
    st.markdown("""
    **How it works:**
    1. Randomly selects a feature and split value
    2. Recursively partitions data using decision trees
    3. Anomalies are isolated with fewer splits (shorter path length)
    4. Anomaly score based on average path length
    
    **When to use:**
    - Large datasets with many features
    - Need for fast, real-time detection
    - When interpretability is less critical
    """)

with st.expander("Autoencoder Explained"):
    st.markdown("""
    **How it works:**
    1. Encoder compresses input into lower-dimensional representation
    2. Decoder reconstructs original input from compressed version
    3. Trained only on normal (non-fraudulent) transactions
    4. High reconstruction error indicates anomaly
    
    **When to use:**
    - Complex, non-linear patterns in data
    - Need for interpretability (via reconstruction errors)
    - When you can isolate normal transaction patterns for training
    """)

with st.expander("Combining with Supervised Methods"):
    st.markdown("""
    Anomaly detection works best when combined with supervised learning:
    
    1. **Anomaly detection** catches unknown fraud patterns
    2. **Supervised models** leverage labeled fraud data
    3. **Ensemble approach** uses both for comprehensive coverage
    
    **Best Practice:**
    - Use anomaly detection as first-line screening
    - Apply supervised models for final decision
    - Combine scores for maximum accuracy
    """)

render_footer()
