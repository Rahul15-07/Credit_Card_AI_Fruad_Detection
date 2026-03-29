import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from footer import render_footer
try:
    from utils.data_processor import DataProcessor
except ImportError:
    from data_processor import DataProcessor
try:
    from utils.model_trainer import ModelTrainer
except ImportError:
    from model_trainer import ModelTrainer
from fraud_predictor import FraudPredictor
from visualization_utils import VisualizationUtils
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="AI Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None

def main():
    st.title("🛡️ AI-Powered Credit Card Fraud Detection System")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Data Analysis", "Model Training", "Model Evaluation", "Real-time Prediction"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Data Analysis":
        show_data_analysis_page()
    elif page == "Model Training":
        show_model_training_page()
    elif page == "Model Evaluation":
        show_model_evaluation_page()
    elif page == "Real-time Prediction":
        show_prediction_page()

def show_home_page():
    st.header("Welcome to the AI Credit Card Fraud Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 System Overview")
        st.write("""
        This system uses advanced Machine Learning algorithms to detect fraudulent 
        credit card transactions in real-time. The system includes:
        
        - **Data Preprocessing Pipeline**: Feature scaling and imbalanced dataset handling
        - **Multiple ML Models**: Logistic Regression, Random Forest, and Neural Networks
        - **Performance Evaluation**: Comprehensive metrics and visualizations
        - **Real-time Prediction**: Interactive fraud detection interface
        """)
        
    with col2:
        st.subheader("🚀 Get Started")
        st.write("""
        1. **Data Analysis**: Explore transaction patterns and fraud distribution
        2. **Model Training**: Train and compare different ML algorithms
        3. **Model Evaluation**: Analyze performance metrics and visualizations
        4. **Real-time Prediction**: Test fraud detection on new transactions
        """)
    
    # Data generation section
    st.markdown("---")
    st.subheader("📈 Generate Sample Dataset")
    st.write("Since real credit card data is sensitive, we'll generate a synthetic dataset for demonstration:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.number_input("Number of Samples", min_value=1000, max_value=50000, value=10000)
    with col2:
        fraud_rate = st.slider("Fraud Rate (%)", min_value=1, max_value=10, value=2)
    with col3:
        n_features = st.number_input("Number of Features", min_value=10, max_value=30, value=20)
    
    if st.button("Generate Dataset", type="primary"):
        with st.spinner("Generating synthetic credit card transaction data..."):
            # Generate synthetic data
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=int(n_features * 0.7),
                n_redundant=int(n_features * 0.2),
                n_clusters_per_class=1,
                weights=[1 - fraud_rate/100, fraud_rate/100],
                flip_y=0.01,
                random_state=42
            )
            
            # Create feature names
            feature_names = [f'V{i+1}' for i in range(n_features)]
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=feature_names)
            df['is_fraud'] = y
            
            # Initialize data processor
            st.session_state.data_processor = DataProcessor()
            st.session_state.data_processor.load_data(df)
            st.session_state.data_loaded = True
            
            st.success(f"Dataset generated successfully! {n_samples} transactions with {fraud_rate}% fraud rate")
            
            # Show basic stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", n_samples)
            with col2:
                st.metric("Fraudulent Transactions", sum(y))
            with col3:
                st.metric("Legitimate Transactions", n_samples - sum(y))
            with col4:
                st.metric("Fraud Rate", f"{(sum(y)/n_samples)*100:.2f}%")

def show_data_analysis_page():
    st.header("📊 Data Analysis & Visualization")
    
    if not st.session_state.data_loaded:
        st.warning("Please generate a dataset first from the Home page.")
        return
    
    data_processor = st.session_state.data_processor
    viz_utils = VisualizationUtils()
    
    # Data overview
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", data_processor.data.shape)
        st.write("**Features:**", len(data_processor.data.columns) - 1)
        
    with col2:
        fraud_count = data_processor.data['is_fraud'].sum()
        total_count = len(data_processor.data)
        st.write("**Fraud Cases:**", fraud_count)
        st.write("**Fraud Rate:**", f"{(fraud_count/total_count)*100:.2f}%")
    
    # Class distribution
    st.subheader("Transaction Distribution")
    fig = viz_utils.plot_class_distribution(data_processor.data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation heatmap
    st.subheader("Feature Correlation Analysis")
    correlation_fig = viz_utils.plot_correlation_heatmap(data_processor.data)
    st.plotly_chart(correlation_fig, use_container_width=True)
    
    # Feature statistics
    st.subheader("Feature Statistics")
    feature_stats = data_processor.data.describe()
    st.dataframe(feature_stats)

def show_model_training_page():
    st.header("🤖 Model Training & Comparison")
    
    if not st.session_state.data_loaded:
        st.warning("Please generate a dataset first from the Home page.")
        return
    
    data_processor = st.session_state.data_processor
    
    st.subheader("Training Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    with col2:
        use_smote = st.checkbox("Use SMOTE for Balancing", value=True)
    with col3:
        random_state = st.number_input("Random State", value=42)
    
    # Model selection
    st.subheader("Model Selection")
    models_to_train = st.multiselect(
        "Select models to train:",
        ["Logistic Regression", "Random Forest", "Neural Network"],
        default=["Logistic Regression", "Random Forest", "Neural Network"]
    )
    
    if st.button("Train Models", type="primary"):
        if not models_to_train:
            st.error("Please select at least one model to train.")
            return
        
        with st.spinner("Training models... This may take a few minutes."):
            # Preprocess data
            X_train, X_test, y_train, y_test = data_processor.preprocess_data(
                test_size=test_size, 
                use_smote=use_smote, 
                random_state=random_state
            )
            
            # Initialize model trainer
            model_trainer = ModelTrainer()
            st.session_state.model_trainer = model_trainer
            
            # Train selected models
            results = {}
            progress_bar = st.progress(0)
            
            for i, model_name in enumerate(models_to_train):
                st.write(f"Training {model_name}...")
                if model_name == "Logistic Regression":
                    results["Logistic Regression"] = model_trainer.train_logistic_regression(
                        X_train, X_test, y_train, y_test
                    )
                elif model_name == "Random Forest":
                    results["Random Forest"] = model_trainer.train_random_forest(
                        X_train, X_test, y_train, y_test
                    )
                elif model_name == "Neural Network":
                    results["Neural Network"] = model_trainer.train_neural_network(
                        X_train, X_test, y_train, y_test
                    )
                
                progress_bar.progress((i + 1) / len(models_to_train))
            
            st.session_state.models_trained = True
            st.session_state.training_results = results
            
            st.success("All models trained successfully!")
            
            # Display training results
            st.subheader("Training Results Summary")
            results_df = pd.DataFrame({
                model: {
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1'],
                    'AUC-ROC': metrics['auc_roc']
                }
                for model, metrics in results.items()
            }).T
            
            st.dataframe(results_df.round(4))

def show_model_evaluation_page():
    st.header("📈 Model Evaluation & Performance Analysis")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the Model Training page.")
        return
    
    results = st.session_state.training_results
    viz_utils = VisualizationUtils()
    
    # Performance comparison
    st.subheader("Model Performance Comparison")
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Metrics comparison
        metrics_fig = viz_utils.plot_model_comparison(results)
        st.plotly_chart(metrics_fig, use_container_width=True)
    
    with col2:
        # ROC curves
        roc_fig = viz_utils.plot_roc_curves(results)
        st.plotly_chart(roc_fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Performance Metrics")
    results_df = pd.DataFrame({
        model: {
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'AUC-ROC': f"{metrics['auc_roc']:.4f}"
        }
        for model, metrics in results.items()
    }).T
    
    st.dataframe(results_df)
    
    # Feature importance (if Random Forest is trained)
    if "Random Forest" in results:
        st.subheader("Feature Importance Analysis")
        model_trainer = st.session_state.model_trainer
        if hasattr(model_trainer, 'rf_model') and model_trainer.rf_model is not None:
            importance_fig = viz_utils.plot_feature_importance(
                model_trainer.rf_model, 
                st.session_state.data_processor.feature_names
            )
            st.plotly_chart(importance_fig, use_container_width=True)
    
    # Model recommendations
    st.subheader("Model Recommendations")
    best_model = max(results.keys(), key=lambda k: results[k]['auc_roc'])
    
    st.success(f"**Best Performing Model:** {best_model}")
    st.write(f"**AUC-ROC Score:** {results[best_model]['auc_roc']:.4f}")
    
    # Performance insights
    st.write("**Performance Insights:**")
    for model, metrics in results.items():
        if metrics['precision'] > 0.9:
            st.write(f"✅ {model}: High precision - Good at avoiding false positives")
        if metrics['recall'] > 0.9:
            st.write(f"✅ {model}: High recall - Good at detecting fraud cases")
        if metrics['f1'] > 0.9:
            st.write(f"✅ {model}: High F1-score - Well-balanced performance")

def show_prediction_page():
    st.header("🔍 Real-time Fraud Prediction")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the Model Training page.")
        return
    
    model_trainer = st.session_state.model_trainer
    data_processor = st.session_state.data_processor
    
    # Model selection for prediction
    available_models = list(st.session_state.training_results.keys())
    selected_model = st.selectbox("Select Model for Prediction:", available_models)
    
    st.subheader("Transaction Details")
    st.write("Enter transaction details below:")
    
    # Create input fields for all features
    feature_inputs = {}
    num_features = len(data_processor.feature_names)
    
    # Create columns for better layout
    cols = st.columns(min(4, num_features))
    
    for i, feature in enumerate(data_processor.feature_names):
        with cols[i % len(cols)]:
            # Use sample statistics for realistic input ranges
            feature_stats = data_processor.data[feature].describe()
            min_val = float(feature_stats['min'])
            max_val = float(feature_stats['max'])
            mean_val = float(feature_stats['mean'])
            
            feature_inputs[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100
            )
    
    # Prediction buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Predict Single Transaction", type="primary"):
            make_prediction(feature_inputs, selected_model, model_trainer, data_processor)
    
    with col2:
        if st.button("Generate Random Transaction"):
            # Generate random values within feature ranges
            for feature in data_processor.feature_names:
                feature_stats = data_processor.data[feature].describe()
                min_val = float(feature_stats['min'])
                max_val = float(feature_stats['max'])
                random_val = np.random.uniform(min_val, max_val)
                feature_inputs[feature] = random_val
            st.rerun()
    
    with col3:
        if st.button("Use Sample Fraud Case"):
            # Use a sample from fraudulent transactions
            fraud_cases = data_processor.data[data_processor.data['is_fraud'] == 1]
            if not fraud_cases.empty:
                sample = fraud_cases.iloc[0]
                for feature in data_processor.feature_names:
                    feature_inputs[feature] = float(sample[feature])
                st.rerun()

def make_prediction(feature_inputs, selected_model, model_trainer, data_processor):
    # Prepare input data
    input_data = pd.DataFrame([feature_inputs])
    
    # Scale the input data using the same scaler used during training
    input_scaled = data_processor.scaler.transform(input_data)
    
    # Get the selected model
    if selected_model == "Logistic Regression":
        model = model_trainer.lr_model
        probabilities = model.predict_proba(input_scaled)[0]
    elif selected_model == "Random Forest":
        model = model_trainer.rf_model
        probabilities = model.predict_proba(input_scaled)[0]
    elif selected_model == "Neural Network":
        model = model_trainer.nn_model
        probabilities = model.predict(input_scaled)[0]
        probabilities = [1 - probabilities[0], probabilities[0]]
    
    prediction = int(probabilities[1] > 0.5)
    confidence = max(probabilities)
    fraud_probability = probabilities[1]
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 1:
            st.error("🚨 **FRAUDULENT TRANSACTION**")
        else:
            st.success("✅ **LEGITIMATE TRANSACTION**")
    
    with col2:
        st.metric("Fraud Probability", f"{fraud_probability:.2%}")
    
    with col3:
        st.metric("Confidence Level", f"{confidence:.2%}")
    
    # Risk assessment
    st.subheader("Risk Assessment")
    
    if fraud_probability >= 0.8:
        risk_level = "Very High"
        risk_color = "🔴"
        recommendation = "Block transaction immediately and contact customer"
    elif fraud_probability >= 0.6:
        risk_level = "High"
        risk_color = "🟠"
        recommendation = "Hold transaction for manual review"
    elif fraud_probability >= 0.4:
        risk_level = "Medium"
        risk_color = "🟡"
        recommendation = "Monitor transaction patterns"
    elif fraud_probability >= 0.2:
        risk_level = "Low"
        risk_color = "🟢"
        recommendation = "Proceed with standard verification"
    else:
        risk_level = "Very Low"
        risk_color = "🟢"
        recommendation = "Process transaction normally"
    
    st.write(f"**Risk Level:** {risk_color} {risk_level}")
    st.write(f"**Recommendation:** {recommendation}")
    
    # Probability visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Legitimate', 'Fraudulent'],
        y=[probabilities[0], probabilities[1]],
        marker_color=['green', 'red'],
        text=[f"{probabilities[0]:.2%}", f"{probabilities[1]:.2%}"],
        textposition='auto'
    ))
    fig.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
    render_footer()
