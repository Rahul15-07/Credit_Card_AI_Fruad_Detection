import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from footer import render_footer
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Model Retraining", page_icon="🔄", layout="wide")

st.title("🔄 Model Retraining Pipeline")
st.markdown("Retrain models with new fraud patterns to maintain detection accuracy")

# Initialize session state for model versions
if 'model_versions' not in st.session_state:
    st.session_state.model_versions = []

if 'retraining_data' not in st.session_state:
    st.session_state.retraining_data = []

# Check if models are trained
if 'models_trained' not in st.session_state or not st.session_state.models_trained:
    st.warning("⚠️ Please train initial models first from the Model Training page.")
    st.stop()

if 'model_trainer' not in st.session_state or st.session_state.model_trainer is None:
    st.error("Model trainer not found. Please retrain models.")
    st.stop()

model_trainer = st.session_state.model_trainer
data_processor = st.session_state.data_processor

st.markdown("---")

# New Data Input Section
st.header("📥 Add New Training Data")

st.info("""
**Why Retrain Models?**
- Fraud patterns evolve over time
- New attack vectors emerge
- Model performance degrades (concept drift)
- Incorporate feedback from false positives/negatives
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Source Options")
    data_source = st.radio(
        "Select data source:",
        ["Upload New Transactions", "Simulated Fraud Patterns", "Feedback from Predictions"],
        help="Choose how to add new training data"
    )

with col2:
    st.subheader("Retraining Strategy")
    retrain_strategy = st.selectbox(
        "Select strategy:",
        ["Incremental Learning (Add to Existing)", "Full Retrain (Replace Data)", "Hybrid (Weighted Mix)"],
        help="How to incorporate new data into models"
    )

st.markdown("---")

# Data Input Methods
new_data = None

if data_source == "Upload New Transactions":
    st.subheader("📂 Upload New Transaction Data")
    
    uploaded_file = st.file_uploader("Upload CSV with labeled transactions", type=['csv'])
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = data_processor.feature_names + ['is_fraud']
            missing_cols = set(required_cols) - set(new_data.columns)
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                new_data = None
            else:
                # Reorder columns and separate features from labels
                X_new = new_data[data_processor.feature_names]
                y_new = new_data['is_fraud']
                
                st.success(f"✅ Loaded {len(new_data)} new transactions")
                st.write(f"**Fraud Distribution:** {sum(y_new)} fraud ({sum(y_new)/len(y_new)*100:.1f}%), {len(y_new)-sum(y_new)} legitimate")
                
                with st.expander("Preview New Data"):
                    st.dataframe(new_data.head(10))
                
                # Store new data
                if st.button("Add to Retraining Queue", type="primary"):
                    st.session_state.retraining_data.append({
                        'X': X_new,
                        'y': y_new,
                        'source': 'uploaded',
                        'timestamp': datetime.now(),
                        'size': len(X_new)
                    })
                    st.success("✅ Data added to retraining queue")
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

elif data_source == "Simulated Fraud Patterns":
    st.subheader("🎲 Generate Simulated Fraud Patterns")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_samples = st.number_input("Number of samples", min_value=10, max_value=1000, value=100)
    
    with col2:
        fraud_ratio = st.slider("Fraud ratio (%)", min_value=1, max_value=50, value=10)
    
    with col3:
        pattern_type = st.selectbox(
            "Fraud pattern",
            ["High Amount Anomalies", "Unusual Time Patterns", "Geographic Anomalies", "Mixed Patterns"]
        )
    
    if st.button("Generate Fraud Patterns", type="primary"):
        # Generate synthetic fraud patterns based on existing data distribution
        num_fraud = int(num_samples * fraud_ratio / 100)
        num_legitimate = num_samples - num_fraud
        
        fraud_data = []
        legitimate_data = []
        
        # Generate fraud transactions with specific patterns
        for _ in range(num_fraud):
            transaction = {}
            
            for i, feature in enumerate(data_processor.feature_names):
                feature_stats = data_processor.data[feature].describe()
                mean = feature_stats['mean']
                std = feature_stats['std']
                
                if pattern_type == "High Amount Anomalies" and i < 3:
                    # Extremely high values for first few features
                    value = np.random.normal(mean + 3*std, std)
                elif pattern_type == "Unusual Time Patterns" and i >= len(data_processor.feature_names) - 5:
                    # Unusual values in later features
                    value = np.random.normal(mean + 2*std, std * 0.5)
                elif pattern_type == "Geographic Anomalies" and 5 <= i < 10:
                    # Anomalous geographic patterns
                    value = np.random.normal(mean - 2*std, std)
                else:
                    # Mixed anomalies
                    if np.random.random() < 0.3:
                        value = np.random.normal(mean + np.random.choice([-2, 2])*std, std)
                    else:
                        value = np.random.normal(mean, std)
                
                transaction[feature] = value
            
            fraud_data.append(transaction)
        
        # Generate legitimate transactions
        for _ in range(num_legitimate):
            transaction = {}
            
            for feature in data_processor.feature_names:
                feature_stats = data_processor.data[feature].describe()
                mean = feature_stats['mean']
                std = feature_stats['std']
                
                value = np.random.normal(mean, std * 0.8)
                transaction[feature] = value
            
            legitimate_data.append(transaction)
        
        # Combine and create labels
        X_new = pd.DataFrame(fraud_data + legitimate_data)
        y_new = pd.Series([1]*num_fraud + [0]*num_legitimate)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_new))
        X_new = X_new.iloc[shuffle_idx].reset_index(drop=True)
        y_new = y_new.iloc[shuffle_idx].reset_index(drop=True)
        
        st.session_state.retraining_data.append({
            'X': X_new,
            'y': y_new,
            'source': f'simulated_{pattern_type}',
            'timestamp': datetime.now(),
            'size': len(X_new)
        })
        
        st.success(f"✅ Generated {num_samples} transactions ({num_fraud} fraud, {num_legitimate} legitimate)")
        st.rerun()

elif data_source == "Feedback from Predictions":
    st.subheader("📊 Incorporate Prediction Feedback")
    
    st.info("""
    Use this to correct misclassifications and improve model accuracy.
    Add transactions that were incorrectly predicted to retrain the models.
    """)
    
    # Manual transaction entry
    st.write("**Enter Transaction Details:**")
    
    transaction_data = {}
    cols_per_row = 4
    
    for i in range(0, len(data_processor.feature_names), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(data_processor.feature_names):
                feature = data_processor.feature_names[i + j]
                feature_stats = data_processor.data[feature].describe()
                
                with col:
                    transaction_data[feature] = st.number_input(
                        f"{feature}",
                        value=float(feature_stats['mean']),
                        format="%.4f",
                        key=f"feedback_{feature}"
                    )
    
    actual_label = st.radio(
        "Actual transaction label:",
        ["Legitimate (0)", "Fraud (1)"],
        horizontal=True
    )
    
    if st.button("Add Feedback Transaction", type="primary"):
        X_new = pd.DataFrame([transaction_data])
        y_new = pd.Series([1 if "Fraud" in actual_label else 0])
        
        st.session_state.retraining_data.append({
            'X': X_new,
            'y': y_new,
            'source': 'feedback',
            'timestamp': datetime.now(),
            'size': 1
        })
        
        st.success("✅ Feedback transaction added to retraining queue")
        st.rerun()

# Retraining Queue Display
st.markdown("---")
st.header("📋 Retraining Queue")

if st.session_state.retraining_data:
    total_samples = sum(item['size'] for item in st.session_state.retraining_data)
    total_fraud = sum(sum(item['y']) for item in st.session_state.retraining_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Queued Batches", len(st.session_state.retraining_data))
    
    with col2:
        st.metric("Total Samples", total_samples)
    
    with col3:
        st.metric("Fraud Cases", int(total_fraud))
    
    with col4:
        fraud_rate = (total_fraud / total_samples * 100) if total_samples > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
    
    # Display queue details
    with st.expander("View Queue Details"):
        queue_data = []
        
        for i, item in enumerate(st.session_state.retraining_data):
            queue_data.append({
                'Batch': i + 1,
                'Source': item['source'],
                'Size': item['size'],
                'Fraud': int(sum(item['y'])),
                'Added': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        st.dataframe(pd.DataFrame(queue_data), use_container_width=True)
    
    # Clear queue option
    if st.button("Clear Queue", type="secondary"):
        st.session_state.retraining_data = []
        st.success("Queue cleared")
        st.rerun()
    
    # Retrain Models Section
    st.markdown("---")
    st.header("🚀 Retrain Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Selection")
        models_to_retrain = st.multiselect(
            "Select models to retrain:",
            list(model_trainer.models.keys()),
            default=list(model_trainer.models.keys())
        )
    
    with col2:
        st.subheader("Retraining Options")
        
        if retrain_strategy == "Hybrid (Weighted Mix)":
            old_data_weight = st.slider(
                "Old data weight",
                min_value=0.1, max_value=0.9, value=0.7, step=0.1,
                help="Higher values give more weight to existing data"
            )
        
        use_smote = st.checkbox(
            "Apply SMOTE",
            value=True,
            help="Balance classes using SMOTE oversampling"
        )
    
    if st.button("🔄 Start Retraining", type="primary"):
        if not models_to_retrain:
            st.error("Please select at least one model to retrain")
        else:
            with st.spinner("Retraining models... This may take a few minutes."):
                # Combine all queued data
                X_new_combined = pd.concat([item['X'] for item in st.session_state.retraining_data])
                y_new_combined = pd.concat([item['y'] for item in st.session_state.retraining_data])
                
                # Get unscaled training data from session state
                if 'X_train_unscaled' not in st.session_state or 'y_train_unscaled' not in st.session_state:
                    st.error("Original unscaled training data not available. Please re-run preprocessing to enable retraining.")
                    st.stop()
                
                X_train_unscaled = st.session_state.X_train_unscaled
                y_train_unscaled = st.session_state.y_train_unscaled
                
                # Prepare data based on strategy
                if retrain_strategy == "Incremental Learning (Add to Existing)":
                    # Combine with existing unscaled training data
                    X_combined = pd.concat([
                        pd.DataFrame(X_train_unscaled, columns=data_processor.feature_names),
                        X_new_combined
                    ]).reset_index(drop=True)
                    
                    y_combined = pd.concat([
                        pd.Series(y_train_unscaled),
                        y_new_combined
                    ]).reset_index(drop=True)
                
                elif retrain_strategy == "Full Retrain (Replace Data)":
                    # Use only new data
                    X_combined = X_new_combined.reset_index(drop=True)
                    y_combined = y_new_combined.reset_index(drop=True)
                
                else:  # Hybrid
                    # Sample from old unscaled data based on weight
                    old_samples = int(len(X_train_unscaled) * old_data_weight)
                    old_indices = np.random.choice(len(X_train_unscaled), old_samples, replace=False)
                    
                    X_old_sample = pd.DataFrame(
                        X_train_unscaled.iloc[old_indices] if hasattr(X_train_unscaled, 'iloc') else X_train_unscaled[old_indices],
                        columns=data_processor.feature_names
                    )
                    y_old_sample = pd.Series(y_train_unscaled.iloc[old_indices] if hasattr(y_train_unscaled, 'iloc') else y_train_unscaled[old_indices])
                    
                    X_combined = pd.concat([X_old_sample, X_new_combined]).reset_index(drop=True)
                    y_combined = pd.concat([y_old_sample, y_new_combined]).reset_index(drop=True)
                
                # Scale the combined unscaled data with a new scaler
                from sklearn.preprocessing import StandardScaler
                new_scaler = StandardScaler()
                X_combined_scaled = new_scaler.fit_transform(X_combined)
                
                # Apply SMOTE on scaled data if requested
                if use_smote:
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=42)
                    X_combined_scaled, y_combined = smote.fit_resample(X_combined_scaled, y_combined)
                
                # Re-scale test data with the new scaler
                X_test_unscaled = st.session_state.X_test_unscaled
                X_test_rescaled = new_scaler.transform(X_test_unscaled)
                
                # Update the scaler in session state
                st.session_state.scaler = new_scaler
                
                # Store old model performance for comparison using OLD scaler
                old_performance = {}
                
                for model_name in models_to_retrain:
                    if model_name in model_trainer.models or model_name in st.session_state.get('models', {}):
                        model = model_trainer.models.get(model_name) or st.session_state.models.get(model_name)
                        y_test = st.session_state.y_test
                        
                        # Use the current X_test (with old scaling) for old performance
                        X_test_old = st.session_state.X_test
                        
                        if model_name == 'Neural Network':
                            y_pred = (model.predict(X_test_old, verbose=0).flatten() > 0.5).astype(int)
                        else:
                            y_pred = model.predict(X_test_old)
                        
                        old_performance[model_name] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, zero_division=0),
                            'recall': recall_score(y_test, y_pred, zero_division=0),
                            'f1': f1_score(y_test, y_pred, zero_division=0)
                        }
                
                # Retrain selected models
                new_performance = {}
                retrained_models = {}
                
                for model_name in models_to_retrain:
                    if model_name == 'Logistic Regression':
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        model.fit(X_combined_scaled, y_combined)
                        retrained_models[model_name] = model
                    
                    elif model_name == 'Random Forest':
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        model.fit(X_combined_scaled, y_combined)
                        retrained_models[model_name] = model
                    
                    elif model_name == 'Neural Network':
                        from tensorflow.keras.models import Sequential
                        from tensorflow.keras.layers import Dense, Dropout
                        
                        model = Sequential([
                            Dense(64, activation='relu', input_shape=(X_combined_scaled.shape[1],)),
                            Dropout(0.3),
                            Dense(32, activation='relu'),
                            Dropout(0.3),
                            Dense(16, activation='relu'),
                            Dense(1, activation='sigmoid')
                        ])
                        
                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                        model.fit(X_combined_scaled, y_combined, epochs=50, batch_size=32, verbose=0, validation_split=0.2)
                        retrained_models[model_name] = model
                    
                    # Evaluate on test set using NEW scaler
                    y_test = st.session_state.y_test
                    
                    if model_name == 'Neural Network':
                        y_pred = (retrained_models[model_name].predict(X_test_rescaled, verbose=0).flatten() > 0.5).astype(int)
                    else:
                        y_pred = retrained_models[model_name].predict(X_test_rescaled)
                    
                    new_performance[model_name] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1': f1_score(y_test, y_pred, zero_division=0)
                    }
                
                # Save current version before updating
                version_info = {
                    'version': len(st.session_state.model_versions) + 1,
                    'timestamp': datetime.now(),
                    'strategy': retrain_strategy,
                    'models': list(models_to_retrain),
                    'old_performance': old_performance,
                    'new_performance': new_performance,
                    'training_samples': len(X_combined),
                    'new_samples_added': len(X_new_combined)
                }
                
                st.session_state.model_versions.append(version_info)
                
                # Update models and test data
                for model_name, model in retrained_models.items():
                    model_trainer.models[model_name] = model
                    if 'models' in st.session_state:
                        st.session_state.models[model_name] = model
                
                # Update test data with new scaling
                st.session_state.X_test = X_test_rescaled
                model_trainer.X_test = X_test_rescaled
                
                # Clear retraining queue
                st.session_state.retraining_data = []
                
                st.success(f"✅ Successfully retrained {len(models_to_retrain)} model(s)!")
                
                # Display performance comparison
                st.subheader("📊 Performance Comparison")
                
                comparison_data = []
                
                for model_name in models_to_retrain:
                    if model_name in old_performance and model_name in new_performance:
                        comparison_data.append({
                            'Model': model_name,
                            'Old Accuracy': old_performance[model_name]['accuracy'],
                            'New Accuracy': new_performance[model_name]['accuracy'],
                            'Accuracy Change': new_performance[model_name]['accuracy'] - old_performance[model_name]['accuracy'],
                            'Old F1': old_performance[model_name]['f1'],
                            'New F1': new_performance[model_name]['f1'],
                            'F1 Change': new_performance[model_name]['f1'] - old_performance[model_name]['f1']
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    st.dataframe(
                        comparison_df.style.format({
                            'Old Accuracy': '{:.4f}',
                            'New Accuracy': '{:.4f}',
                            'Accuracy Change': '{:+.4f}',
                            'Old F1': '{:.4f}',
                            'New F1': '{:.4f}',
                            'F1 Change': '{:+.4f}'
                        }).background_gradient(subset=['Accuracy Change', 'F1 Change'], cmap='RdYlGn', vmin=-0.1, vmax=0.1),
                        use_container_width=True
                    )
                    
                    # Visualize improvements
                    fig = go.Figure()
                    
                    for metric in ['Accuracy', 'F1']:
                        fig.add_trace(go.Bar(
                            name=f'Old {metric}',
                            x=[d['Model'] for d in comparison_data],
                            y=[d[f'Old {metric}'] for d in comparison_data],
                            text=[f"{d[f'Old {metric}']:.3f}" for d in comparison_data],
                            textposition='auto'
                        ))
                        
                        fig.add_trace(go.Bar(
                            name=f'New {metric}',
                            x=[d['Model'] for d in comparison_data],
                            y=[d[f'New {metric}'] for d in comparison_data],
                            text=[f"{d[f'New {metric}']:.3f}" for d in comparison_data],
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title='Model Performance: Before vs After Retraining',
                        xaxis_title='Model',
                        yaxis_title='Score',
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.rerun()

else:
    st.info("👆 Add new training data to the retraining queue to get started")

# Version History
st.markdown("---")
st.header("📜 Model Version History")

if st.session_state.model_versions:
    st.write(f"**Total Versions:** {len(st.session_state.model_versions)}")
    
    # Display version timeline
    for i, version in enumerate(reversed(st.session_state.model_versions)):
        with st.expander(
            f"Version {version['version']} - {version['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} "
            f"({'Latest' if i == 0 else f'{i} version(s) ago'})"
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Retraining Details:**")
                st.write(f"- Strategy: {version['strategy']}")
                st.write(f"- Models Updated: {', '.join(version['models'])}")
                st.write(f"- Training Samples: {version['training_samples']}")
                st.write(f"- New Samples Added: {version['new_samples_added']}")
            
            with col2:
                st.write("**Performance Summary:**")
                
                for model_name in version['models']:
                    if model_name in version['new_performance']:
                        new_perf = version['new_performance'][model_name]
                        
                        if model_name in version['old_performance']:
                            old_perf = version['old_performance'][model_name]
                            acc_change = new_perf['accuracy'] - old_perf['accuracy']
                            change_emoji = "📈" if acc_change > 0 else "📉" if acc_change < 0 else "➡️"
                            
                            st.write(f"{model_name}: {new_perf['accuracy']:.4f} {change_emoji} ({acc_change:+.4f})")
                        else:
                            st.write(f"{model_name}: {new_perf['accuracy']:.4f}")
else:
    st.info("No retraining history yet. Retrain models to start tracking versions.")

# Best Practices Section
st.markdown("---")
st.header("💡 Retraining Best Practices")

with st.expander("When to Retrain"):
    st.markdown("""
    **Retrain your models when:**
    
    1. **Performance Degradation**
       - Accuracy drops below acceptable threshold
       - Increasing false positives or false negatives
       - F1-score declines significantly
    
    2. **New Fraud Patterns Emerge**
       - Novel attack vectors detected
       - Changes in customer behavior
       - New payment methods introduced
    
    3. **Seasonal Changes**
       - Holiday shopping patterns
       - End-of-year spending
       - Special events or promotions
    
    4. **Regular Schedule**
       - Weekly: For high-volume systems
       - Monthly: For moderate-volume systems
       - Quarterly: For stable, low-volume systems
    """)

with st.expander("Retraining Strategies"):
    st.markdown("""
    **Choose the right strategy:**
    
    1. **Incremental Learning**
       - **Use when:** Adding small amounts of new data
       - **Pros:** Preserves historical knowledge, fast retraining
       - **Cons:** May accumulate outdated patterns
    
    2. **Full Retrain**
       - **Use when:** Data distribution has changed significantly
       - **Pros:** Clean slate, adapts quickly to new patterns
       - **Cons:** Loses historical knowledge, requires more data
    
    3. **Hybrid (Weighted Mix)**
       - **Use when:** Balancing old and new patterns
       - **Pros:** Best of both worlds, controlled adaptation
       - **Cons:** Requires tuning weight parameter
    """)

with st.expander("Data Quality Tips"):
    st.markdown("""
    **Ensure high-quality retraining data:**
    
    1. **Label Accuracy**
       - Verify fraud labels with investigation results
       - Review edge cases with domain experts
       - Correct mislabeled transactions
    
    2. **Data Balance**
       - Maintain appropriate fraud ratio
       - Use SMOTE for imbalanced datasets
       - Avoid over-representing rare cases
    
    3. **Data Freshness**
       - Prioritize recent fraud patterns
       - Remove outdated attack methods
       - Weight recent data higher
    
    4. **Feature Consistency**
       - Ensure same features as original training
       - Validate data quality and ranges
       - Handle missing values appropriately
    """)

render_footer()
