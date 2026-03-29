import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from footer import render_footer
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

st.set_page_config(page_title="Real-Time Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Real-Time Fraud Prediction")
st.markdown("Enter transaction details to get real-time fraud predictions from trained models.")

# Check if models are trained
if not st.session_state.get('models_trained', False):
    st.warning("⚠️ Please train models first.")
    st.stop()

trained_models = st.session_state.trained_models
scaler = st.session_state.get('scaler', None)

st.markdown("---")

# Transaction Input Form
st.subheader("💳 Transaction Details")

# Create input form
with st.form("transaction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Basic Information**")
        time_input = st.number_input("Time (seconds from first transaction)", min_value=0.0, value=3600.0)
        amount_input = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0, step=0.01)
        
        st.markdown("**Transaction Features V1-V10**")
        v1 = st.number_input("V1", value=0.0, step=0.1, format="%.3f")
        v2 = st.number_input("V2", value=0.0, step=0.1, format="%.3f")
        v3 = st.number_input("V3", value=0.0, step=0.1, format="%.3f")
        v4 = st.number_input("V4", value=0.0, step=0.1, format="%.3f")
        v5 = st.number_input("V5", value=0.0, step=0.1, format="%.3f")
        v6 = st.number_input("V6", value=0.0, step=0.1, format="%.3f")
        v7 = st.number_input("V7", value=0.0, step=0.1, format="%.3f")
        v8 = st.number_input("V8", value=0.0, step=0.1, format="%.3f")
        v9 = st.number_input("V9", value=0.0, step=0.1, format="%.3f")
        v10 = st.number_input("V10", value=0.0, step=0.1, format="%.3f")
    
    with col2:
        st.markdown("**Transaction Features V11-V20**")
        v11 = st.number_input("V11", value=0.0, step=0.1, format="%.3f")
        v12 = st.number_input("V12", value=0.0, step=0.1, format="%.3f")
        v13 = st.number_input("V13", value=0.0, step=0.1, format="%.3f")
        v14 = st.number_input("V14", value=0.0, step=0.1, format="%.3f")
        v15 = st.number_input("V15", value=0.0, step=0.1, format="%.3f")
        v16 = st.number_input("V16", value=0.0, step=0.1, format="%.3f")
        v17 = st.number_input("V17", value=0.0, step=0.1, format="%.3f")
        v18 = st.number_input("V18", value=0.0, step=0.1, format="%.3f")
        v19 = st.number_input("V19", value=0.0, step=0.1, format="%.3f")
        v20 = st.number_input("V20", value=0.0, step=0.1, format="%.3f")
    
    with col3:
        st.markdown("**Transaction Features V21-V28**")
        v21 = st.number_input("V21", value=0.0, step=0.1, format="%.3f")
        v22 = st.number_input("V22", value=0.0, step=0.1, format="%.3f")
        v23 = st.number_input("V23", value=0.0, step=0.1, format="%.3f")
        v24 = st.number_input("V24", value=0.0, step=0.1, format="%.3f")
        v25 = st.number_input("V25", value=0.0, step=0.1, format="%.3f")
        v26 = st.number_input("V26", value=0.0, step=0.1, format="%.3f")
        v27 = st.number_input("V27", value=0.0, step=0.1, format="%.3f")
        v28 = st.number_input("V28", value=0.0, step=0.1, format="%.3f")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.form_submit_button("🔍 Predict Fraud Risk", type="primary")

# Quick test samples
st.markdown("---")
st.subheader("🎯 Quick Test Samples")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Sample: Low Risk Transaction"):
        # Typical legitimate transaction pattern
        st.session_state.update({
            'sample_time': 3600.0,
            'sample_amount': 50.0,
            'sample_v1': -0.5, 'sample_v2': 0.2, 'sample_v3': -0.1,
            'sample_v4': 0.3, 'sample_v5': -0.2, 'sample_v6': 0.1
        })

with col2:
    if st.button("Sample: Medium Risk Transaction"):
        # Moderately suspicious pattern
        st.session_state.update({
            'sample_time': 7200.0,
            'sample_amount': 500.0,
            'sample_v1': -1.2, 'sample_v2': 1.8, 'sample_v3': -2.1,
            'sample_v4': 1.5, 'sample_v5': -1.8, 'sample_v6': 2.2
        })

with col3:
    if st.button("Sample: High Risk Transaction"):
        # Highly suspicious pattern
        st.session_state.update({
            'sample_time': 10800.0,
            'sample_amount': 2000.0,
            'sample_v1': -3.5, 'sample_v2': 4.2, 'sample_v3': -5.1,
            'sample_v4': 3.8, 'sample_v5': -4.5, 'sample_v6': 5.5
        })

# Make predictions
if predict_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'Time': [time_input],
        'V1': [v1], 'V2': [v2], 'V3': [v3], 'V4': [v4], 'V5': [v5],
        'V6': [v6], 'V7': [v7], 'V8': [v8], 'V9': [v9], 'V10': [v10],
        'V11': [v11], 'V12': [v12], 'V13': [v13], 'V14': [v14], 'V15': [v15],
        'V16': [v16], 'V17': [v17], 'V18': [v18], 'V19': [v19], 'V20': [v20],
        'V21': [v21], 'V22': [v22], 'V23': [v23], 'V24': [v24], 'V25': [v25],
        'V26': [v26], 'V27': [v27], 'V28': [v28], 'Amount': [amount_input]
    })
    
    # Apply scaling if scaler exists
    if scaler is not None:
        # Reorder columns to exactly match the order the scaler was fitted on
        if hasattr(scaler, 'feature_names_in_'):
            fitted_columns = list(scaler.feature_names_in_)
            for col in fitted_columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[fitted_columns]
        input_data_scaled = pd.DataFrame(
            scaler.transform(input_data),
            columns=input_data.columns
        )
    else:
        input_data_scaled = input_data
    
    # Make predictions with all models
    predictions = {}
    probabilities = {}
    
    for model_name, model in trained_models.items():
        try:
            if model_name == 'deep_nn':
                # TensorFlow model
                prob = model.predict(input_data_scaled, verbose=0)[0][0]
                pred = 1 if prob > 0.5 else 0
            else:
                # Scikit-learn model
                pred = model.predict(input_data_scaled)[0]
                prob = model.predict_proba(input_data_scaled)[0][1]
            
            predictions[model_name] = pred
            probabilities[model_name] = prob
            
        except Exception as e:
            st.error(f"Error making prediction with {model_name}: {str(e)}")
    
    # Display results
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    # Overall risk assessment
    avg_probability = np.mean(list(probabilities.values()))
    fraud_votes = sum(predictions.values())
    total_models = len(predictions)
    
    # Risk level determination
    if avg_probability < 0.3:
        risk_level = "LOW"
        risk_color = "green"
        risk_emoji = "🟢"
    elif avg_probability < 0.7:
        risk_level = "MEDIUM"
        risk_color = "orange"
        risk_emoji = "🟡"
    else:
        risk_level = "HIGH"
        risk_color = "red"
        risk_emoji = "🔴"
    
    # Display overall assessment
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Risk Level", f"{risk_emoji} {risk_level}")
    with col2:
        st.metric("Average Probability", f"{avg_probability:.2%}")
    with col3:
        st.metric("Models Flagged", f"{fraud_votes}/{total_models}")
    with col4:
        if risk_level == "HIGH":
            st.error("⚠️ BLOCK TRANSACTION")
        elif risk_level == "MEDIUM":
            st.warning("🔍 REVIEW REQUIRED")
        else:
            st.success("✅ APPROVE TRANSACTION")
    
    # Detailed model results
    st.subheader("🔍 Detailed Model Results")
    
    results_data = []
    for model_name in trained_models.keys():
        prediction = predictions[model_name]
        probability = probabilities[model_name]
        confidence = probability if prediction == 1 else (1 - probability)
        
        results_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'Fraud Probability': f"{probability:.2%}",
            'Confidence': f"{confidence:.2%}"
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Probability visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of probabilities
        fig = px.bar(
            x=[name.replace('_', ' ').title() for name in probabilities.keys()],
            y=list(probabilities.values()),
            title="Fraud Probability by Model",
            color=list(probabilities.values()),
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(showlegend=False)
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Decision Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gauge chart for average probability
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Average Fraud Probability"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgreen"},
                    {'range': [0.3, 0.7], 'color': "yellow"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Transaction details summary
    st.subheader("💳 Transaction Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Transaction Details**:
        - Amount: ${amount_input:,.2f}
        - Time: {time_input:,.0f} seconds
        - Risk Level: {risk_emoji} {risk_level}
        - Fraud Probability: {avg_probability:.2%}
        """)
    
    with col2:
        st.markdown(f"""
        **Decision Recommendation**:
        - Models in Agreement: {fraud_votes}/{total_models}
        - Confidence Level: {np.mean([probabilities[m] if predictions[m] == 1 else (1-probabilities[m]) for m in probabilities]):.2%}
        - Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)
    
    # Store prediction history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    prediction_record = {
        'timestamp': datetime.now().isoformat(),
        'amount': amount_input,
        'risk_level': risk_level,
        'avg_probability': avg_probability,
        'predictions': predictions,
        'probabilities': probabilities
    }
    
    st.session_state.prediction_history.append(prediction_record)
    
    # Feature importance (for tree-based models)
    if 'random_forest' in trained_models:
        st.subheader("🌳 Feature Importance Analysis")
        
        rf_model = trained_models['random_forest']
        feature_names = input_data.columns.tolist()
        importance = rf_model.feature_importances_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        # Plot feature importance
        fig = px.bar(
            importance_df.tail(15),  # Top 15 features
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 15 Feature Importance (Random Forest)",
            color='Importance',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

# Prediction history
if st.session_state.get('prediction_history'):
    st.markdown("---")
    st.subheader("📊 Recent Prediction History")
    
    # Display recent predictions
    history = st.session_state.prediction_history[-10:]  # Last 10 predictions
    
    history_df = pd.DataFrame([
        {
            'Timestamp': record['timestamp'],
            'Amount': f"${record['amount']:,.2f}",
            'Risk Level': record['risk_level'],
            'Fraud Probability': f"{record['avg_probability']:.2%}"
        }
        for record in reversed(history)
    ])
    
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.rerun()

# Tips and guidelines
st.markdown("---")
st.subheader("💡 Usage Guidelines")

with st.expander("Feature Descriptions"):
    st.markdown("""
    - **Time**: Time elapsed since the first transaction in the dataset
    - **Amount**: Transaction amount in USD
    - **V1-V28**: Principal components obtained with PCA transformation
    - These features represent various transaction characteristics that have been anonymized for privacy
    """)

with st.expander("Risk Level Interpretation"):
    st.markdown("""
    - **LOW Risk (🟢)**: Probability < 30% - Approve transaction
    - **MEDIUM Risk (🟡)**: Probability 30-70% - Manual review recommended
    - **HIGH Risk (🔴)**: Probability > 70% - Block transaction
    
    Consider the business cost of false positives vs false negatives when setting thresholds.
    """)

with st.expander("Model Ensemble Approach"):
    st.markdown("""
    - Multiple models provide different perspectives on the same transaction
    - Consensus among models increases confidence in the prediction
    - Individual model strengths can complement each other
    - Use the average probability for final decision making
    """)

render_footer()
