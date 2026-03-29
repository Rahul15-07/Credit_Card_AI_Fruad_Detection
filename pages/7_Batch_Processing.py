import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from footer import render_footer
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

st.set_page_config(page_title="Batch Processing", page_icon="📦", layout="wide")

st.title("📦 Batch Transaction Processing")
st.markdown("Analyze multiple transactions simultaneously for efficient fraud detection")

# Check if models are trained
if 'models_trained' not in st.session_state or not st.session_state.models_trained:
    st.warning("⚠️ Please train models first from the Model Training page.")
    st.stop()

if 'model_trainer' not in st.session_state or st.session_state.model_trainer is None:
    st.error("Model trainer not found. Please retrain models.")
    st.stop()

model_trainer = st.session_state.model_trainer
data_processor = st.session_state.data_processor

st.markdown("---")

# Input Method Selection
st.header("📥 Select Input Method")

input_method = st.radio(
    "How would you like to provide transaction data?",
    ["Upload CSV File", "Generate Sample Batch", "Use Test Data"],
    horizontal=True
)

batch_data = None

if input_method == "Upload CSV File":
    st.subheader("Upload Transaction CSV")
    
    st.info("""
    **CSV Format Requirements:**
    - File must include all feature columns used in training
    - Feature names must match exactly
    - No 'is_fraud' column should be included (predictions will be added)
    - Example features: feature_1, feature_2, ... feature_N
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(batch_data)} transactions")
            
            # Show preview
            with st.expander("Preview Data"):
                st.dataframe(batch_data.head(10))
            
            # Validate and reorder columns
            expected_features = data_processor.feature_names
            missing_features = set(expected_features) - set(batch_data.columns)
            
            if missing_features:
                st.error(f"Missing required features: {missing_features}")
                batch_data = None
            else:
                # Reorder columns to match expected feature order and drop extra columns
                batch_data = batch_data[expected_features]
                st.success("✅ All required features found and aligned correctly")
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

elif input_method == "Generate Sample Batch":
    st.subheader("Generate Sample Transaction Batch")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_samples = st.number_input(
            "Number of Transactions",
            min_value=10, max_value=1000, value=100, step=10
        )
    
    with col2:
        sample_type = st.selectbox(
            "Sample Type",
            ["Random Mix", "Mostly Normal", "Mostly Suspicious", "High Risk"]
        )
    
    if st.button("Generate Batch", type="primary"):
        # Generate synthetic transactions based on training data distribution
        batch_transactions = []
        
        for _ in range(num_samples):
            transaction = {}
            
            for feature in data_processor.feature_names:
                feature_stats = data_processor.data[feature].describe()
                mean = feature_stats['mean']
                std = feature_stats['std']
                
                if sample_type == "Random Mix":
                    value = np.random.normal(mean, std)
                elif sample_type == "Mostly Normal":
                    value = np.random.normal(mean, std * 0.5)
                elif sample_type == "Mostly Suspicious":
                    value = np.random.normal(mean, std * 1.5)
                else:  # High Risk
                    value = np.random.normal(mean, std * 2)
                
                transaction[feature] = value
            
            batch_transactions.append(transaction)
        
        batch_data = pd.DataFrame(batch_transactions)
        st.success(f"✅ Generated {num_samples} sample transactions")
        
        with st.expander("Preview Generated Data"):
            st.dataframe(batch_data.head(10))

else:  # Use Test Data
    st.subheader("Use Test Dataset")
    
    if model_trainer.X_test is not None:
        num_samples = st.slider(
            "Number of test transactions to use",
            min_value=10, max_value=min(500, len(model_trainer.X_test)),
            value=min(100, len(model_trainer.X_test))
        )
        
        # Get sample from test set
        indices = np.random.choice(len(model_trainer.X_test), num_samples, replace=False)
        batch_data = pd.DataFrame(
            model_trainer.X_test[indices],
            columns=data_processor.feature_names
        )
        
        # Store actual labels for comparison
        st.session_state.batch_actual_labels = model_trainer.y_test.iloc[indices].values if hasattr(model_trainer.y_test, 'iloc') else model_trainer.y_test[indices]
        
        st.success(f"✅ Using {num_samples} transactions from test set")
        
        with st.expander("Preview Test Data"):
            st.dataframe(batch_data.head(10))
    else:
        st.error("Test data not available. Please train models first.")

st.markdown("---")

# Model Selection and Processing
if batch_data is not None:
    st.header("🤖 Model Selection & Processing")
    
    # Get available models
    available_models = list(model_trainer.models.keys())
    
    # Add ensemble models if available
    if 'ensemble_model' in st.session_state and st.session_state.ensemble_model.ensemble_results:
        available_models.extend(['Voting Ensemble', 'Stacking Ensemble', 'Weighted Ensemble'])
    
    # Add anomaly detection models if available
    if 'anomaly_detector' in st.session_state and st.session_state.anomaly_detector.results:
        available_models.extend(['Isolation Forest', 'Autoencoder'])
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_models = st.multiselect(
            "Select models for batch prediction:",
            available_models,
            default=available_models[:min(3, len(available_models))]
        )
    
    with col2:
        st.write("**Processing Options:**")
        include_probabilities = st.checkbox("Include Probability Scores", value=True)
        include_risk_levels = st.checkbox("Include Risk Levels", value=True)
    
    if st.button("🚀 Process Batch", type="primary"):
        if not selected_models:
            st.error("Please select at least one model")
        else:
            with st.spinner("Processing batch predictions..."):
                # Scale the input data
                batch_scaled = data_processor.scaler.transform(batch_data)
                
                # Store predictions from all models
                all_predictions = {}
                all_probabilities = {}
                
                for model_name in selected_models:
                    try:
                        if model_name in model_trainer.models:
                            model = model_trainer.models[model_name]
                            
                            if model_name == 'Neural Network':
                                proba = model.predict(batch_scaled, verbose=0).flatten()
                                pred = (proba > 0.5).astype(int)
                            else:
                                pred = model.predict(batch_scaled)
                                proba = model.predict_proba(batch_scaled)[:, 1]
                            
                            all_predictions[model_name] = pred
                            all_probabilities[model_name] = proba
                        
                        elif 'ensemble' in model_name.lower():
                            ensemble_model = st.session_state.ensemble_model
                            
                            if 'Voting' in model_name and ensemble_model.voting_model:
                                pred, proba = ensemble_model.predict_voting(batch_scaled)
                                all_predictions[model_name] = pred
                                all_probabilities[model_name] = proba
                            elif 'Stacking' in model_name and ensemble_model.stacking_model:
                                pred, proba = ensemble_model.predict_stacking(batch_scaled)
                                all_predictions[model_name] = pred
                                all_probabilities[model_name] = proba
                            elif 'Weighted' in model_name and ensemble_model.weighted_model:
                                pred, proba = ensemble_model.predict_weighted(batch_scaled)
                                all_predictions[model_name] = pred
                                all_probabilities[model_name] = proba
                        
                        elif model_name in ['Isolation Forest', 'Autoencoder']:
                            anomaly_detector = st.session_state.anomaly_detector
                            
                            if model_name == 'Isolation Forest' and anomaly_detector.isolation_forest:
                                pred, proba = anomaly_detector.predict_isolation_forest(batch_scaled)
                                all_predictions[model_name] = pred
                                all_probabilities[model_name] = proba
                            elif model_name == 'Autoencoder' and anomaly_detector.autoencoder:
                                pred, proba = anomaly_detector.predict_autoencoder(batch_scaled)
                                all_predictions[model_name] = pred
                                all_probabilities[model_name] = proba
                    
                    except Exception as e:
                        st.warning(f"Could not process {model_name}: {str(e)}")
                
                # Create results DataFrame
                results_df = batch_data.copy()
                results_df['Transaction_ID'] = range(1, len(results_df) + 1)
                
                # Add predictions and probabilities
                for model_name in all_predictions.keys():
                    results_df[f'{model_name}_Prediction'] = all_predictions[model_name]
                    
                    if include_probabilities:
                        results_df[f'{model_name}_Probability'] = all_probabilities[model_name]
                    
                    if include_risk_levels:
                        # Calculate risk levels
                        risk_levels = []
                        for prob in all_probabilities[model_name]:
                            if prob >= 0.8:
                                risk_levels.append('Very High')
                            elif prob >= 0.6:
                                risk_levels.append('High')
                            elif prob >= 0.4:
                                risk_levels.append('Medium')
                            elif prob >= 0.2:
                                risk_levels.append('Low')
                            else:
                                risk_levels.append('Very Low')
                        
                        results_df[f'{model_name}_Risk_Level'] = risk_levels
                
                # Calculate consensus prediction (majority vote)
                if len(all_predictions) > 1:
                    predictions_array = np.array(list(all_predictions.values())).T
                    consensus = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 1, predictions_array)
                    results_df['Consensus_Prediction'] = consensus
                    
                    # Average probability
                    probabilities_array = np.array(list(all_probabilities.values())).T
                    avg_probability = np.mean(probabilities_array, axis=1)
                    results_df['Average_Probability'] = avg_probability
                
                # Store results in session state
                st.session_state.batch_results = results_df
                
                st.success(f"✅ Batch processing complete! Processed {len(results_df)} transactions")

# Display Results
if 'batch_results' in st.session_state:
    st.markdown("---")
    st.header("📊 Batch Processing Results")
    
    results_df = st.session_state.batch_results
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    
    # Count predictions for each model
    prediction_cols = [col for col in results_df.columns if '_Prediction' in col]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_transactions = len(results_df)
        st.metric("Total Transactions", total_transactions)
    
    with col2:
        if 'Consensus_Prediction' in results_df.columns:
            fraud_detected = sum(results_df['Consensus_Prediction'])
            st.metric("Fraud Detected (Consensus)", fraud_detected)
        else:
            fraud_detected = sum(results_df[prediction_cols[0]])
            st.metric("Fraud Detected", fraud_detected)
    
    with col3:
        fraud_rate = (fraud_detected / total_transactions) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    with col4:
        if 'Average_Probability' in results_df.columns:
            avg_risk = results_df['Average_Probability'].mean()
            st.metric("Average Fraud Probability", f"{avg_risk:.2%}")
    
    # Model Agreement Analysis
    if len(prediction_cols) > 1:
        st.markdown("---")
        st.subheader("Model Agreement Analysis")
        
        # Calculate agreement percentage
        predictions_array = results_df[prediction_cols].values
        agreement = np.all(predictions_array == predictions_array[:, 0:1], axis=1)
        agreement_pct = (sum(agreement) / len(agreement)) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Agreement", f"{agreement_pct:.1f}%", 
                     help="Percentage of transactions where all models agree")
        
        with col2:
            disagreement_count = len(agreement) - sum(agreement)
            st.metric("Disagreements", disagreement_count,
                     help="Transactions where models disagree - require manual review")
    
    # Risk Distribution
    st.markdown("---")
    st.subheader("Risk Distribution")
    
    if 'Average_Probability' in results_df.columns:
        # Create histogram of fraud probabilities
        fig = px.histogram(
            results_df, 
            x='Average_Probability',
            nbins=50,
            title='Distribution of Fraud Probabilities',
            labels={'Average_Probability': 'Fraud Probability', 'count': 'Number of Transactions'},
            color_discrete_sequence=['#1f77b4']
        )
        
        # Add risk threshold lines
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                     annotation_text="Classification Threshold")
        fig.add_vline(x=0.8, line_dash="dash", line_color="red",
                     annotation_text="High Risk")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Results Table
    st.markdown("---")
    st.subheader("Detailed Results")
    
    # Filtering options
    col1, col2 = st.columns(2)
    
    with col1:
        filter_option = st.selectbox(
            "Filter Results:",
            ["Show All", "Fraud Only", "Legitimate Only", "High Risk Only", "Disagreements Only"]
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort By:",
            ["Transaction_ID"] + [col for col in results_df.columns if 'Probability' in col]
        )
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if filter_option == "Fraud Only":
        if 'Consensus_Prediction' in results_df.columns:
            filtered_df = filtered_df[filtered_df['Consensus_Prediction'] == 1]
        else:
            filtered_df = filtered_df[filtered_df[prediction_cols[0]] == 1]
    
    elif filter_option == "Legitimate Only":
        if 'Consensus_Prediction' in results_df.columns:
            filtered_df = filtered_df[filtered_df['Consensus_Prediction'] == 0]
        else:
            filtered_df = filtered_df[filtered_df[prediction_cols[0]] == 0]
    
    elif filter_option == "High Risk Only":
        if 'Average_Probability' in results_df.columns:
            filtered_df = filtered_df[filtered_df['Average_Probability'] >= 0.7]
    
    elif filter_option == "Disagreements Only" and len(prediction_cols) > 1:
        predictions_array = filtered_df[prediction_cols].values
        agreement = np.all(predictions_array == predictions_array[:, 0:1], axis=1)
        filtered_df = filtered_df[~agreement]
    
    # Sort
    filtered_df = filtered_df.sort_values(sort_by, ascending=False)
    
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Export Options
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"fraud_detection_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )
    
    with col2:
        # JSON Export
        json_str = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download as JSON",
            data=json_str,
            file_name=f"fraud_detection_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # Summary Report
        risk_distribution_text = ""
        
        if 'Average_Probability' in filtered_df.columns:
            risk_distribution_text = f"""
Risk Distribution:
- Very High: {len(filtered_df[filtered_df['Average_Probability'] >= 0.8])}
- High: {len(filtered_df[(filtered_df['Average_Probability'] >= 0.6) & (filtered_df['Average_Probability'] < 0.8)])}
- Medium: {len(filtered_df[(filtered_df['Average_Probability'] >= 0.4) & (filtered_df['Average_Probability'] < 0.6)])}
- Low: {len(filtered_df[(filtered_df['Average_Probability'] >= 0.2) & (filtered_df['Average_Probability'] < 0.4)])}
- Very Low: {len(filtered_df[filtered_df['Average_Probability'] < 0.2])}"""
        
        summary_report = f"""
Fraud Detection Batch Processing Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Statistics:
- Total Transactions: {len(results_df)}
- Fraud Detected: {fraud_detected}
- Fraud Rate: {fraud_rate:.2f}%

Models Used: {', '.join(prediction_cols).replace('_Prediction', '')}
{risk_distribution_text}
        """
        
        st.download_button(
            label="Download Summary Report",
            data=summary_report,
            file_name=f"fraud_detection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    # Performance Comparison (if actual labels available)
    if 'batch_actual_labels' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Performance on Test Data")
        
        actual_labels = st.session_state.batch_actual_labels
        
        # Calculate metrics for each model
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        performance_data = []
        
        for col in prediction_cols:
            model_name = col.replace('_Prediction', '')
            predictions = results_df[col].values
            
            acc = accuracy_score(actual_labels, predictions)
            prec = precision_score(actual_labels, predictions, zero_division=0)
            rec = recall_score(actual_labels, predictions, zero_division=0)
            f1 = f1_score(actual_labels, predictions, zero_division=0)
            
            performance_data.append({
                'Model': model_name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1
            })
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df.style.highlight_max(axis=0, color='lightgreen'))

# Information Section
st.markdown("---")
st.header("ℹ️ Batch Processing Guide")

with st.expander("How to Use Batch Processing"):
    st.markdown("""
    **Step 1: Prepare Your Data**
    - CSV file with all required features
    - Or generate sample data for testing
    - Or use existing test data
    
    **Step 2: Select Models**
    - Choose one or more models for prediction
    - Multiple models provide consensus predictions
    
    **Step 3: Process Batch**
    - Click "Process Batch" to run predictions
    - View results and statistics
    
    **Step 4: Export Results**
    - Download in CSV, JSON, or text format
    - Use results for further analysis
    """)

with st.expander("Understanding Results"):
    st.markdown("""
    **Consensus Prediction:**
    - Majority vote from all selected models
    - More reliable than single model
    
    **Average Probability:**
    - Mean fraud probability across all models
    - Use for risk scoring
    
    **Risk Levels:**
    - Very High (≥80%): Block transaction
    - High (60-80%): Manual review required
    - Medium (40-60%): Monitor closely
    - Low (20-40%): Standard verification
    - Very Low (<20%): Process normally
    
    **Model Disagreements:**
    - Transactions where models disagree
    - Require manual review
    - Indicate edge cases or complex patterns
    """)

with st.expander("Best Practices"):
    st.markdown("""
    1. **Use Multiple Models:**
       - Ensemble provides more robust predictions
       - Reduces false positives and false negatives
    
    2. **Review Disagreements:**
       - Always manually review transactions where models disagree
       - These often represent complex fraud patterns
    
    3. **Monitor Performance:**
       - Track fraud rates over time
       - Update models with new fraud patterns
    
    4. **Set Appropriate Thresholds:**
       - Balance false positives vs false negatives
       - Consider business impact of each error type
    
    5. **Regular Updates:**
       - Retrain models periodically
       - Incorporate new fraud patterns
       - Adjust thresholds based on performance
    """)

render_footer()
