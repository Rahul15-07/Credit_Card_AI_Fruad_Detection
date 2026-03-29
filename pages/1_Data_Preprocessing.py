import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from footer import render_footer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processor import DataProcessor

st.set_page_config(page_title="Data Preprocessing", page_icon="🔧", layout="wide")

st.title("🔧 Data Preprocessing Pipeline")
st.markdown("Clean, transform, and prepare your transaction data for machine learning models.")

# Check if data is loaded
if not st.session_state.get('data_loaded', False) or st.session_state.data_processor is None:
    st.warning("⚠️ Please upload your dataset on the main page first.")
    st.stop()

data_processor = st.session_state.data_processor
original_data = data_processor.data.copy()

st.markdown("---")

# Data Overview
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", len(original_data))
with col2:
    st.metric("Features", len(original_data.columns) - 1)
with col3:
    if 'Class' in original_data.columns:
        fraud_count = original_data['Class'].sum()
        st.metric("Fraudulent Transactions", fraud_count)
with col4:
    if 'Class' in original_data.columns:
        fraud_rate = original_data['Class'].sum() / len(original_data) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

# Class distribution visualization
if 'Class' in original_data.columns:
    st.subheader("📈 Class Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        class_counts = original_data['Class'].value_counts()
        fig = px.bar(
            x=['Legitimate', 'Fraudulent'], 
            y=class_counts.values,
            title="Transaction Class Distribution",
            color=['Legitimate', 'Fraudulent'],
            color_discrete_map={'Legitimate': 'lightblue', 'Fraudulent': 'red'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            values=class_counts.values,
            names=['Legitimate', 'Fraudulent'],
            title="Class Distribution (Percentage)",
            color_discrete_map={'Legitimate': 'lightblue', 'Fraudulent': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Preprocessing Options
st.subheader("⚙️ Preprocessing Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Feature Scaling**")
    scaling_method = st.selectbox(
        "Select Scaling Method",
        ["StandardScaler", "RobustScaler", "None"],
        help="StandardScaler: Zero mean, unit variance. RobustScaler: Uses median and IQR."
    )
    
    st.markdown("**Data Splitting**")
    test_size = st.slider(
        "Test Set Size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Proportion of data to use for testing"
    )
    
    random_state = st.number_input(
        "Random State",
        value=42,
        help="Seed for reproducible results"
    )

with col2:
    st.markdown("**Imbalanced Data Handling**")
    resampling_method = st.selectbox(
        "Select Resampling Method",
        ["None", "SMOTE", "Random Undersampling", "SMOTEENN"],
        help="Methods to handle class imbalance"
    )
    
    if resampling_method == "SMOTE":
        smote_k_neighbors = st.slider(
            "SMOTE K-Neighbors",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of neighbors for SMOTE algorithm"
        )
    
    if resampling_method == "Random Undersampling":
        sampling_strategy = st.slider(
            "Sampling Strategy",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            help="Ratio of majority to minority class after resampling"
        )

# Process button
if st.button("🚀 Start Preprocessing", type="primary"):
    with st.spinner("Processing data..."):
        try:
            # Prepare features and target
            if 'is_fraud' in original_data.columns:
                X = original_data.drop('is_fraud', axis=1)
                y = original_data['is_fraud']
            else:
                st.error("Target column 'is_fraud' not found in dataset")
                st.stop()
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Store unscaled versions and original labels for retraining
            st.session_state.X_train_unscaled = X_train.copy()
            st.session_state.X_test_unscaled = X_test.copy()
            st.session_state.y_train_unscaled = y_train.copy()
            
            # Feature scaling
            if scaling_method != "None":
                if scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                elif scaling_method == "RobustScaler":
                    scaler = RobustScaler()
                
                X_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
                
                X_train = X_train_scaled
                X_test = X_test_scaled
                
                # Store scaler for future use
                st.session_state.scaler = scaler
            else:
                st.session_state.scaler = None
            
            # Handle imbalanced data
            if resampling_method != "None":
                if resampling_method == "SMOTE":
                    resampler = SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)
                elif resampling_method == "Random Undersampling":
                    resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
                elif resampling_method == "SMOTEENN":
                    resampler = SMOTEENN(random_state=random_state)
                
                X_train, y_train = resampler.fit_resample(X_train, y_train)
            
            # Store processed data
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.preprocessing_done = True
            
            st.success("✅ Preprocessing completed successfully!")
            
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")

# Display results if preprocessing is done
if st.session_state.get('preprocessing_done', False):
    st.markdown("---")
    st.subheader("📋 Preprocessing Results")
    
    # Data shapes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", len(st.session_state.X_train))
    with col2:
        st.metric("Test Samples", len(st.session_state.X_test))
    with col3:
        train_fraud_rate = st.session_state.y_train.sum() / len(st.session_state.y_train) * 100
        st.metric("Train Fraud Rate", f"{train_fraud_rate:.2f}%")
    with col4:
        test_fraud_rate = st.session_state.y_test.sum() / len(st.session_state.y_test) * 100
        st.metric("Test Fraud Rate", f"{test_fraud_rate:.2f}%")
    
    # Class distribution after preprocessing
    col1, col2 = st.columns(2)
    
    with col1:
        train_counts = st.session_state.y_train.value_counts()
        fig = px.bar(
            x=['Legitimate', 'Fraudulent'], 
            y=train_counts.values,
            title="Training Set Class Distribution (After Preprocessing)",
            color=['Legitimate', 'Fraudulent'],
            color_discrete_map={'Legitimate': 'lightblue', 'Fraudulent': 'red'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        test_counts = st.session_state.y_test.value_counts()
        fig = px.bar(
            x=['Legitimate', 'Fraudulent'], 
            y=test_counts.values,
            title="Test Set Class Distribution",
            color=['Legitimate', 'Fraudulent'],
            color_discrete_map={'Legitimate': 'lightblue', 'Fraudulent': 'red'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature statistics
    st.subheader("📊 Feature Statistics")
    
    # Select features to display
    selected_features = st.multiselect(
        "Select features to view statistics",
        st.session_state.X_train.columns.tolist(),
        default=st.session_state.X_train.columns.tolist()[:5]
    )
    
    if selected_features:
        stats_df = st.session_state.X_train[selected_features].describe()
        st.dataframe(stats_df)
    
    st.info("🎯 Data preprocessing complete! Navigate to 'Model Training' to build ML models.")

# Tips and information
st.markdown("---")
st.subheader("💡 Preprocessing Tips")

with st.expander("Feature Scaling"):
    st.markdown("""
    - **StandardScaler**: Best for normally distributed features
    - **RobustScaler**: Better for features with outliers
    - Use scaling when features have different ranges (e.g., Amount vs Time)
    """)

with st.expander("Handling Imbalanced Data"):
    st.markdown("""
    - **SMOTE**: Creates synthetic minority samples
    - **Random Undersampling**: Reduces majority class samples
    - **SMOTEENN**: Combines oversampling and undersampling
    - Choose based on your dataset size and requirements
    """)

with st.expander("Data Splitting"):
    st.markdown("""
    - Use stratified splitting to maintain class distribution
    - Typical split: 80% training, 20% testing
    - Set random state for reproducible results
    """)

render_footer()
