import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from footer import render_footer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
import tensorflow as tf
from tensorflow import keras
import pickle
import os
from types import SimpleNamespace
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Model Training")
st.markdown("Train and compare multiple ML algorithms for fraud detection.")

# Check if preprocessing is done
if not st.session_state.get('preprocessing_done', False):
    st.warning("⚠️ Please complete data preprocessing first.")
    st.stop()

# ── Always ensure model_trainer is a real writable object ────────────────────
# Use SimpleNamespace so we never depend on an external class import.
# If a stale None is in session state (from a failed previous run), replace it.
if st.session_state.get('model_trainer') is None:
    st.session_state.model_trainer = SimpleNamespace(
        models={},
        X_train=None, X_test=None,
        y_train=None, y_test=None
    )
model_trainer = st.session_state.model_trainer

st.markdown("---")

# Training Configuration
st.subheader("⚙️ Training Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Select Models to Train**")
    train_logistic      = st.checkbox("Logistic Regression", value=True)
    train_random_forest = st.checkbox("Random Forest", value=True)
    train_mlp           = st.checkbox("Neural Network (MLP)", value=True)
    train_deep_nn       = st.checkbox("Deep Neural Network (TensorFlow)", value=False)

with col2:
    st.markdown("**Cross-Validation**")
    use_cv   = st.checkbox("Use Cross-Validation", value=True,
                           help="Validate models using k-fold CV")
    cv_folds = 5
    if use_cv:
        cv_folds = st.slider("CV Folds", min_value=3, max_value=10, value=5)

# ── Hyperparameter defaults (used if expanders are hidden) ───────────────────
lr_C = 1.0;     lr_solver = "liblinear"; lr_penalty = "l2"; lr_max_iter = 1000
rf_n_estimators = 100; rf_max_depth = 10
rf_min_samples_split = 2; rf_min_samples_leaf = 1
mlp_hidden_layers = "100,50"; mlp_activation = "relu"
mlp_solver = "adam";           mlp_max_iter = 500
dnn_layers = "128,64,32";      dnn_dropout = 0.3
dnn_batch_size = 32;           dnn_epochs = 50

st.subheader("🔧 Hyperparameter Configuration")

if train_logistic:
    with st.expander("Logistic Regression Parameters"):
        c1, c2 = st.columns(2)
        with c1:
            lr_C      = st.slider("Regularization (C)", 0.001, 100.0, 1.0, 0.001)
            lr_solver = st.selectbox("Solver",
                ["liblinear", "lbfgs", "newton-cg", "sag", "saga"])
        with c2:
            lr_penalty  = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])
            lr_max_iter = st.number_input("Max Iterations", 100, 10000, 1000)

if train_random_forest:
    with st.expander("Random Forest Parameters"):
        c1, c2 = st.columns(2)
        with c1:
            rf_n_estimators = st.slider("Number of Trees", 10, 500, 100)
            rf_max_depth    = st.slider("Max Depth", 3, 50, 10)
        with c2:
            rf_min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
            rf_min_samples_leaf  = st.slider("Min Samples Leaf",  1, 10, 1)

if train_mlp:
    with st.expander("Neural Network (MLP) Parameters"):
        c1, c2 = st.columns(2)
        with c1:
            mlp_hidden_layers = st.text_input("Hidden Layer Sizes", value="100,50",
                                              help="Comma-separated layer sizes")
            mlp_activation = st.selectbox("Activation Function",
                ["relu", "tanh", "logistic"])
        with c2:
            mlp_solver   = st.selectbox("Solver", ["adam", "lbfgs", "sgd"])
            mlp_max_iter = st.number_input("Max Iterations", 100, 5000, 500)

if train_deep_nn:
    with st.expander("Deep Neural Network Parameters"):
        c1, c2 = st.columns(2)
        with c1:
            dnn_layers   = st.text_input("Layer Configuration", value="128,64,32",
                                         help="Comma-separated layer sizes")
            dnn_dropout  = st.slider("Dropout Rate", 0.0, 0.8, 0.3, 0.1)
        with c2:
            dnn_batch_size = st.selectbox("Batch Size", [32, 64, 128, 256])
            dnn_epochs     = st.slider("Epochs", 10, 200, 50)

# ── Training ──────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🚀 Start Training", type="primary"):

    models_to_train = []
    if train_logistic:      models_to_train.append('logistic')
    if train_random_forest: models_to_train.append('random_forest')
    if train_mlp:           models_to_train.append('mlp')
    if train_deep_nn:       models_to_train.append('deep_nn')

    if not models_to_train:
        st.error("Please select at least one model to train.")
        st.stop()

    progress_bar = st.progress(0)
    status_text  = st.empty()
    trained_models  = {}
    training_scores = {}

    for i, model_type in enumerate(models_to_train):
        status_text.text(f"Training {model_type.replace('_', ' ').title()}...")
        progress_bar.progress(i / len(models_to_train))

        try:
            # ── Build model ───────────────────────────────────────────────
            if model_type == 'logistic':
                _solver  = lr_solver
                _penalty = lr_penalty if lr_penalty != 'none' else None
                if lr_penalty == 'l1' and _solver not in ['liblinear', 'saga']:
                    _solver = 'liblinear'
                elif lr_penalty == 'elasticnet' and _solver != 'saga':
                    _solver = 'saga'
                model = LogisticRegression(
                    C=lr_C, solver=_solver, penalty=_penalty,
                    max_iter=lr_max_iter, random_state=42
                )

            elif model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                    min_samples_split=rf_min_samples_split,
                    min_samples_leaf=rf_min_samples_leaf,
                    random_state=42, n_jobs=-1
                )

            elif model_type == 'mlp':
                hidden_sizes = tuple(
                    int(x) for x in mlp_hidden_layers.split(',') if x.strip()
                )
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_sizes,
                    activation=mlp_activation, solver=mlp_solver,
                    max_iter=mlp_max_iter, random_state=42
                )

            elif model_type == 'deep_nn':
                layer_sizes = [int(x) for x in dnn_layers.split(',') if x.strip()]
                nn = keras.Sequential()
                nn.add(keras.layers.Dense(
                    layer_sizes[0], activation='relu',
                    input_shape=(st.session_state.X_train.shape[1],)))
                nn.add(keras.layers.Dropout(dnn_dropout))
                for ls in layer_sizes[1:]:
                    nn.add(keras.layers.Dense(ls, activation='relu'))
                    nn.add(keras.layers.Dropout(dnn_dropout))
                nn.add(keras.layers.Dense(1, activation='sigmoid'))
                nn.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['accuracy'])
                model = nn

            # ── Train model ───────────────────────────────────────────────
            if model_type == 'deep_nn':
                history = model.fit(
                    st.session_state.X_train, st.session_state.y_train,
                    epochs=dnn_epochs, batch_size=dnn_batch_size,
                    validation_split=0.2, verbose=0
                )
                training_scores[model_type] = {
                    'train_accuracy': history.history['accuracy'][-1],
                    'val_accuracy':   history.history['val_accuracy'][-1],
                    'train_loss':     history.history['loss'][-1],
                    'val_loss':       history.history['val_loss'][-1],
                }
            else:
                model.fit(st.session_state.X_train, st.session_state.y_train)
                if use_cv:
                    cv_sc = cross_val_score(
                        model, st.session_state.X_train, st.session_state.y_train,
                        cv=cv_folds, scoring='accuracy'
                    )
                    training_scores[model_type] = {
                        'cv_mean': cv_sc.mean(), 'cv_std': cv_sc.std(),
                        'cv_scores': cv_sc,
                    }
                else:
                    training_scores[model_type] = {
                        'train_accuracy': model.score(
                            st.session_state.X_train, st.session_state.y_train)
                    }

            trained_models[model_type] = model

        except Exception as e:
            st.error(f"Error training {model_type}: {str(e)}")

    progress_bar.progress(1.0)
    status_text.text("Training completed!")

    # ── Persist basic state ───────────────────────────────────────────────
    st.session_state.trained_models  = trained_models
    st.session_state.training_scores = training_scores
    st.session_state.models_trained  = True

    # Human-readable display names
    _name_map = {
        'logistic':      'Logistic Regression',
        'random_forest': 'Random Forest',
        'mlp':           'Neural Network',
        'deep_nn':       'Deep Neural Network',
    }

    # ── Build a fresh SimpleNamespace — always writable, no import needed ─
    new_mt = SimpleNamespace(
        models    = {_name_map.get(k, k): v for k, v in trained_models.items()},
        X_train   = st.session_state.X_train,
        X_test    = st.session_state.X_test,
        y_train   = st.session_state.y_train,
        y_test    = st.session_state.y_test,
    )
    st.session_state.model_trainer = new_mt
    model_trainer = new_mt   # refresh local reference

    # ── Compute test-set metrics for Evaluation / Ensemble pages ─────────
    _results = {}
    for _key, _model in trained_models.items():
        _display = _name_map.get(_key, _key)
        try:
            if _key == 'deep_nn':
                _proba = _model.predict(st.session_state.X_test, verbose=0).flatten()
                _pred  = (_proba > 0.5).astype(int)
            else:
                _pred  = _model.predict(st.session_state.X_test)
                _proba = _model.predict_proba(st.session_state.X_test)[:, 1]

            _results[_display] = {
                'accuracy':         accuracy_score(st.session_state.y_test, _pred),
                'precision':        precision_score(st.session_state.y_test, _pred,
                                                    zero_division=0),
                'recall':           recall_score(st.session_state.y_test, _pred,
                                                 zero_division=0),
                'f1':               f1_score(st.session_state.y_test, _pred,
                                             zero_division=0),
                'auc_roc':          roc_auc_score(st.session_state.y_test, _proba),
                'confusion_matrix': confusion_matrix(st.session_state.y_test, _pred),
                'roc_data':         roc_curve(st.session_state.y_test, _proba),
            }
        except Exception as _e:
            st.warning(f"Could not compute metrics for {_display}: {_e}")
    st.session_state.training_results = _results

    # ── Save models to disk ───────────────────────────────────────────────
    try:
        os.makedirs('models', exist_ok=True)
        for model_name, model in trained_models.items():
            if model_name == 'deep_nn':
                model.save(f'models/{model_name}_model.h5')
            else:
                with open(f'models/{model_name}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
    except Exception as e:
        st.warning(f"Could not save models to disk: {e}")

    st.success("✅ All models trained successfully!")

# ── Display training results ──────────────────────────────────────────────────
if st.session_state.get('models_trained', False) and \
   'training_scores' in st.session_state:

    st.markdown("---")
    st.subheader("📊 Training Results")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Training Accuracy Summary**")
        accuracy_data = []
        for model_name, scores in st.session_state.training_scores.items():
            label = model_name.replace('_', ' ').title()
            if 'cv_mean' in scores:
                accuracy_data.append({
                    'Model': label,
                    'Accuracy': f"{scores['cv_mean']:.4f} ± {scores['cv_std']:.4f}"
                })
            elif 'train_accuracy' in scores:
                accuracy_data.append({'Model': label,
                                      'Accuracy': f"{scores['train_accuracy']:.4f}"})
            elif 'val_accuracy' in scores:
                accuracy_data.append({'Model': label,
                                      'Accuracy': f"{scores['val_accuracy']:.4f}"})
        st.dataframe(pd.DataFrame(accuracy_data),
                     use_container_width=True, hide_index=True)

    with col2:
        fig_data = []
        for model_name, scores in st.session_state.training_scores.items():
            val = (scores.get('cv_mean') or
                   scores.get('train_accuracy') or
                   scores.get('val_accuracy'))
            if val is not None:
                fig_data.append({'Model': model_name.replace('_', ' ').title(),
                                 'Accuracy': val})
        if fig_data:
            fig_df = pd.DataFrame(fig_data)
            fig = px.bar(fig_df, x='Model', y='Accuracy',
                         title="Model Training Accuracy Comparison",
                         color='Accuracy', color_continuous_scale='viridis')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    if use_cv:
        st.subheader("🔍 Cross-Validation Details")
        cv_models = [n for n, s in st.session_state.training_scores.items()
                     if 'cv_scores' in s]
        if cv_models:
            selected_cv_model = st.selectbox("Select model for CV details", cv_models)
            if selected_cv_model:
                cv_scores = st.session_state.training_scores[selected_cv_model]['cv_scores']
                col1, col2 = st.columns(2)
                with col1:
                    cv_df = pd.DataFrame({
                        'Fold': range(1, len(cv_scores) + 1),
                        'Accuracy': cv_scores
                    })
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
                with col2:
                    fig = px.line(cv_df, x='Fold', y='Accuracy',
                                  title=f"CV Scores — "
                                        f"{selected_cv_model.replace('_', ' ').title()}",
                                  markers=True)
                    fig.add_hline(y=cv_scores.mean(), line_dash="dash",
                                  line_color="red",
                                  annotation_text=f"Mean: {cv_scores.mean():.4f}")
                    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏗️ Model Information")
    model_info = []
    for model_name in st.session_state.trained_models:
        info = {'Model': model_name.replace('_', ' ').title()}
        if model_name == 'logistic':
            info['Parameters'] = f"C={lr_C}, Solver={lr_solver}"
        elif model_name == 'random_forest':
            info['Parameters'] = f"Trees={rf_n_estimators}, Max Depth={rf_max_depth}"
        elif model_name == 'mlp':
            info['Parameters'] = (f"Layers={mlp_hidden_layers}, "
                                   f"Activation={mlp_activation}")
        elif model_name == 'deep_nn':
            info['Parameters'] = f"Layers={dnn_layers}, Dropout={dnn_dropout}"
        model_info.append(info)
    st.dataframe(pd.DataFrame(model_info), use_container_width=True, hide_index=True)

    st.info("🎯 Model training complete! Navigate to 'Model Evaluation' to assess performance.")

# ── Tips ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💡 Training Tips")

with st.expander("Model Selection Guidelines"):
    st.markdown("""
    - **Logistic Regression**: Fast, interpretable, good baseline
    - **Random Forest**: Handles non-linear patterns, robust to outliers
    - **Neural Networks**: Can capture complex patterns, may need more data
    - **Deep Networks**: For very large datasets with complex patterns
    """)

with st.expander("Hyperparameter Tuning"):
    st.markdown("""
    - Start with default parameters and adjust based on performance
    - Use cross-validation to get reliable performance estimates
    - Consider regularization to prevent overfitting
    - Monitor training time vs performance trade-offs
    """)

with st.expander("Performance Optimization"):
    st.markdown("""
    - Use feature scaling for distance-based algorithms
    - Handle class imbalance appropriately
    - Consider ensemble methods for better performance
    - Save trained models for future use
    """)

render_footer()
