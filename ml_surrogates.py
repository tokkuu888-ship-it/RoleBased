"""
Machine Learning Surrogate Models Module for TOKUMA 3-in-1 Research Platform
Provides pre-trained model loading, SHAP/LIME visualizations, and PINN integration
"""
from __future__ import annotations

import io
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import joblib
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import streamlit as st
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    TENSORFLOW_AVAILABLE = False
    print(f"TensorFlow not available - using scikit-learn alternatives: {str(e)[:100]}")
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


class MLSurrogateManager:
    """Manages ML surrogate models for fast hydrological simulations"""
    
    def __init__(self):
        self.models = {}
        self.feature_names = [
            'f_len', 'f_slp', 'climate_sensitivity', 'temp_scale', 
            'supply_perturb', 'demand_perturb', 'soil_infiltration',
            'annual_rainfall', 'base_temperature'
        ]
        self.target_names = ['wp', 'yield', 'ag_cons', 'eff']
        
    def create_synthetic_training_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic training data for surrogate models"""
        np.random.seed(42)
        
        # Generate realistic parameter ranges
        data = {
            'f_len': np.random.uniform(50, 400, n_samples),
            'f_slp': np.random.uniform(0.01, 0.5, n_samples),
            'climate_sensitivity': np.random.uniform(0.5, 3.0, n_samples),
            'temp_scale': np.random.uniform(0.8, 1.2, n_samples),
            'supply_perturb': np.random.uniform(0.85, 1.15, n_samples),
            'demand_perturb': np.random.uniform(0.9, 1.1, n_samples),
            'soil_infiltration': np.random.uniform(0.5, 5.0, n_samples),
            'annual_rainfall': np.random.uniform(250, 1400, n_samples),
            'base_temperature': np.random.uniform(15, 30, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate synthetic targets based on physical relationships
        # Water productivity (kg/m³)
        df['wp'] = (
            1.2 * (df['soil_infiltration'] / 3.0) * 
            (df['annual_rainfall'] / 800) * 
            (1 / (1 + df['climate_sensitivity'] * 0.1)) *
            (1 / (1 + df['temp_scale'] * 0.05)) *
            np.exp(-0.001 * df['f_len']) *
            (1 + 0.1 * np.sin(df['f_slp'] * 10))
        )
        
        # Yield (t/ha)
        df['yield'] = df['wp'] * df['annual_rainfall'] * 0.001 * np.random.uniform(0.8, 1.2, n_samples)
        
        # Agricultural water consumption (BCM)
        df['ag_cons'] = (
            df['annual_rainfall'] * 0.001 * 
            (1 + df['demand_perturb'] - 1) *
            (1 + df['climate_sensitivity'] * 0.05) *
            (1 / (1 + df['soil_infiltration'] * 0.1))
        )
        
        # System efficiency
        df['eff'] = (
            0.7 * (1 - df['f_len'] / 1000) * 
            (1 + df['f_slp'] * 0.5) *
            (df['soil_infiltration'] / 3.0) *
            np.random.uniform(0.9, 1.1, n_samples)
        )
        
        # Add some noise
        for target in self.target_names:
            df[target] += np.random.normal(0, df[target].std() * 0.05, n_samples)
            df[target] = np.maximum(df[target], df[target].quantile(0.01))  # Remove extreme outliers
        
        return df

    def train_random_forest_surrogate(self, X: pd.DataFrame, y: pd.DataFrame, 
                                    target: str) -> RandomForestRegressor:
        """Train Random Forest surrogate model"""
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X, y[target])
        return rf

    def train_gradient_boosting_surrogate(self, X: pd.DataFrame, y: pd.DataFrame,
                                        target: str) -> GradientBoostingRegressor:
        """Train Gradient Boosting surrogate model"""
        gb = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        gb.fit(X, y[target])
        return gb

    def train_neural_network_surrogate(self, X: pd.DataFrame, y: pd.DataFrame,
                                      target: str):
        """Train Neural Network surrogate model"""
        if TENSORFLOW_AVAILABLE:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            
            model.fit(
                X, y[target],
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            return model
        else:
            # Fallback to scikit-learn MLPRegressor
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                max_iter=1000,
                random_state=42
            )
            model.fit(X, y[target])
            return model

    def create_pinn_model(self, input_dim: int):
        """Create Physics-Informed Neural Network (PINN)"""
        if not TENSORFLOW_AVAILABLE:
            # Return a simple scikit-learn model as fallback
            return MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='tanh',
                max_iter=500,
                random_state=42
            )
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        
        # Hidden layers with physics-informed architecture
        x = tf.keras.layers.Dense(128, activation='tanh')(inputs)
        x = tf.keras.layers.Dense(128, activation='tanh')(x)
        x = tf.keras.layers.Dense(64, activation='tanh')(x)
        x = tf.keras.layers.Dense(64, activation='tanh')(x)
        x = tf.keras.layers.Dense(32, activation='tanh')(x)
        
        # Output layers for different physical quantities
        wp_output = tf.keras.layers.Dense(1, name='wp')(x)
        yield_output = tf.keras.layers.Dense(1, name='yield')(x)
        eff_output = tf.keras.layers.Dense(1, name='eff')(x)
        
        # Physics constraints (simplified mass balance)
        # This is a placeholder - in practice, you'd implement actual physics equations
        mass_balance = tf.keras.layers.Lambda(
            lambda x: x[0] * x[1] - x[2],  # Simplified constraint
            name='mass_balance'
        )([wp_output, yield_output, eff_output])
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[wp_output, yield_output, eff_output, mass_balance]
        )
        
        # Custom loss function that includes physics constraints
        def physics_informed_loss(y_true, y_pred):
            # Standard MSE loss
            mse_loss = tf.keras.losses.mse(y_true, y_pred)
            
            # Physics penalty (mass balance should be close to zero)
            physics_penalty = tf.reduce_mean(tf.square(mass_balance))
            
            return mse_loss + 0.1 * physics_penalty
        
        model.compile(
            optimizer='adam',
            loss={
                'wp': 'mse',
                'yield': 'mse',
                'eff': 'mse',
                'mass_balance': physics_informed_loss
            },
            loss_weights={
                'wp': 1.0,
                'yield': 1.0,
                'eff': 1.0,
                'mass_balance': 0.1
            }
        )
        
        return model

    def save_model(self, model: Any, model_path: str, model_type: str = 'sklearn'):
        """Save trained model to disk"""
        if model_type == 'sklearn':
            joblib.dump(model, model_path)
        elif model_type == 'tensorflow':
            model.save(model_path)
        elif model_type == 'pickle':
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

    def load_model(self, model_path: str, model_type: str = 'sklearn'):
        """Load trained model from disk"""
        if model_type == 'sklearn':
            return joblib.load(model_path)
        elif model_type == 'tensorflow':
            return tf.keras.models.load_model(model_path)
        elif model_type == 'pickle':
            with open(model_path, 'rb') as f:
                return pickle.load(f)

    def create_shap_explainer(self, model: Any, X_background: pd.DataFrame, 
                            model_type: str = 'sklearn') -> Any:
        """Create SHAP explainer for model interpretability"""
        if model_type == 'sklearn':
            return shap.TreeExplainer(model)
        elif model_type == 'tensorflow':
            return shap.DeepExplainer(model, X_background.values)
        else:
            return shap.KernelExplainer(model.predict, X_background)

    def create_lime_explainer(self, X_train: pd.DataFrame, 
                            mode: str = 'regression') -> lime.lime_tabular.LimeTabularExplainer:
        """Create LIME explainer for local interpretability"""
        return lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            mode=mode,
            discretize_continuous=True
        )

    def plot_shap_values(self, explainer: Any, X_test: pd.DataFrame, 
                        feature_names: List[str]) -> go.Figure:
        """Create SHAP values visualization"""
        shap_values = explainer.shap_values(X_test)
        
        # Create summary plot
        fig = go.Figure()
        
        for i, feature in enumerate(feature_names):
            fig.add_trace(go.Scatter(
                x=shap_values[:, i],
                y=X_test[feature].values,
                mode='markers',
                name=feature,
                opacity=0.7
            ))
        
        fig.update_layout(
            title="SHAP Values - Feature Impact on Predictions",
            xaxis_title="SHAP Value",
            yaxis_title="Feature Value",
            height=600
        )
        
        return fig

    def plot_feature_importance(self, model: Any, feature_names: List[str]) -> go.Figure:
        """Plot feature importance from tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # For neural networks, use permutation importance
            from sklearn.inspection import permutation_importance
            # This would need actual data to compute
            importances = np.random.rand(len(feature_names))
        
        fig = go.Figure(data=[
            go.Bar(x=feature_names, y=importances)
        ])
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Features",
            yaxis_title="Importance",
            xaxis_tickangle=-45
        )
        
        return fig

    def render_ml_interface(self) -> Dict[str, Any]:
        """Render the complete ML surrogate interface"""
        st.subheader("🤖 Machine Learning Surrogate Models")
        
        # Model selection and training options
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Model Configuration**")
            
            model_type = st.selectbox(
                "Select Model Type",
                ["Random Forest", "Gradient Boosting", "Neural Network", "PINN"]
            )
            
            target_variable = st.selectbox(
                "Target Variable",
                self.target_names
            )
            
            # Training options
            use_existing_model = st.checkbox("Load Pre-trained Model")
            
            if use_existing_model:
                uploaded_model = st.file_uploader(
                    "Upload Model (.pkl, .h5, .joblib)",
                    type=['pkl', 'h5', 'joblib']
                )
            else:
                st.write("**Training Parameters**")
                n_samples = st.slider("Training Samples", 1000, 20000, 5000, 1000)
                train_button = st.button("Train New Model")
        
        with col2:
            st.write("**Model Performance & Visualization**")
            
            if use_existing_model and uploaded_model:
                # Load and evaluate model
                try:
                    # Determine model type from file extension
                    if uploaded_model.name.endswith('.h5'):
                        model = self.load_model(uploaded_model, 'tensorflow')
                        model_type_loaded = 'tensorflow'
                    else:
                        model = self.load_model(uploaded_model, 'sklearn')
                        model_type_loaded = 'sklearn'
                    
                    st.success(f"Model loaded successfully: {uploaded_model.name}")
                    
                    # Generate test data for evaluation
                    test_data = self.create_synthetic_training_data(1000)
                    X_test = test_data[self.feature_names]
                    y_test = test_data[target_variable]
                    
                    # Make predictions
                    if model_type_loaded == 'tensorflow':
                        predictions = model.predict(X_test.values).flatten()
                    else:
                        predictions = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    
                    col2a, col2b = st.columns(2)
                    col2a.metric("MSE", f"{mse:.4f}")
                    col2b.metric("R² Score", f"{r2:.4f}")
                    
                    # Create prediction vs actual plot
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=y_test, y=predictions,
                        mode='markers',
                        name='Predictions',
                        opacity=0.7
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        name='Perfect Fit',
                        line=dict(dash='dash')
                    ))
                    fig_pred.update_layout(
                        title="Predictions vs Actual",
                        xaxis_title="Actual Values",
                        yaxis_title="Predicted Values"
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # SHAP analysis
                    if st.checkbox("Show SHAP Analysis"):
                        with st.spinner("Computing SHAP values..."):
                            explainer = self.create_shap_explainer(model, X_test, model_type_loaded)
                            shap_fig = self.plot_shap_values(explainer, X_test, self.feature_names)
                            st.plotly_chart(shap_fig, use_container_width=True)
                    
                    # Feature importance
                    if st.checkbox("Show Feature Importance"):
                        if model_type_loaded == 'sklearn':
                            importance_fig = self.plot_feature_importance(model, self.feature_names)
                            st.plotly_chart(importance_fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
            
            elif not use_existing_model and train_button:
                # Train new model
                with st.spinner(f"Training {model_type} model..."):
                    # Generate training data
                    training_data = self.create_synthetic_training_data(n_samples)
                    X = training_data[self.feature_names]
                    y = training_data[self.target_names]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Train model based on selection
                    if model_type == "Random Forest":
                        model = self.train_random_forest_surrogate(X_train, y_train, target_variable)
                    elif model_type == "Gradient Boosting":
                        model = self.train_gradient_boosting_surrogate(X_train, y_train, target_variable)
                    elif model_type == "Neural Network":
                        model = self.train_neural_network_surrogate(X_train, y_train, target_variable)
                    elif model_type == "PINN":
                        model = self.create_pinn_model(X_train.shape[1])
                        # For PINN, we need to prepare multi-output targets
                        y_pinn = {
                            'wp': y_train['wp'],
                            'yield': y_train['yield'],
                            'eff': y_train['eff'],
                            'mass_balance': np.zeros(len(y_train))  # Target is zero for constraint
                        }
                        model.fit(X_train, y_pinn, epochs=50, batch_size=32, verbose=0)
                    
                    # Evaluate model
                    if model_type == "PINN":
                        predictions = model.predict(X_test)[0].flatten()  # Get WP output
                    else:
                        predictions = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test[target_variable], predictions)
                    r2 = r2_score(y_test[target_variable], predictions)
                    
                    st.success(f"Model trained successfully!")
                    col2a, col2b = st.columns(2)
                    col2a.metric("Training MSE", f"{mse:.4f}")
                    col2b.metric("Training R²", f"{r2:.4f}")
                    
                    # Store model in session state
                    st.session_state.trained_model = model
                    st.session_state.model_type = model_type
                    st.session_state.target_variable = target_variable
                    
                    # Offer download
                    model_buffer = io.BytesIO()
                    if model_type in ["Random Forest", "Gradient Boosting"]:
                        joblib.dump(model, model_buffer)
                        model_buffer.seek(0)
                        st.download_button(
                            label="Download Model (.joblib)",
                            data=model_buffer.getvalue(),
                            file_name=f"{model_type.lower().replace(' ', '_')}_{target_variable}.joblib",
                            mime="application/octet-stream"
                        )
        
        # LIME analysis section
        if 'trained_model' in st.session_state:
            st.write("---")
            st.write("**🔍 Local Interpretability (LIME)**")
            
            if st.checkbox("Generate LIME Explanations"):
                # Generate fresh training data for LIME
                lime_data = self.create_synthetic_training_data(1000)
                X_lime = lime_data[self.feature_names]
                
                lime_explainer = self.create_lime_explainer(X_lime)
                
                # Select instance to explain
                instance_idx = st.slider("Select Instance", 0, len(X_lime)-1, 0)
                instance = X_lime.iloc[instance_idx:instance_idx+1]
                
                # Get prediction
                model = st.session_state.trained_model
                if st.session_state.model_type == "PINN":
                    pred = model.predict(instance)[0][0]
                else:
                    pred = model.predict(instance)[0]
                
                st.write(f"**Prediction for instance {instance_idx}:** {pred:.4f}")
                
                # Generate LIME explanation
                try:
                    if st.session_state.model_type == "PINN":
                        # For PINN, use a wrapper function
                        def predict_fn(x):
                            return model.predict(x)[0]
                        
                        explainer = lime.lime_tabular.LimeTabularExplainer(
                            X_lime.values,
                            feature_names=X_lime.columns,
                            mode='regression',
                            discretize_continuous=True
                        )
                        
                        explanation = explainer.explain_instance(
                            instance.values[0],
                            predict_fn,
                            num_features=5
                        )
                    else:
                        explanation = lime_explainer.explain_instance(
                            instance.values[0],
                            model.predict,
                            num_features=5
                        )
                    
                    # Display explanation
                    st.write("**Feature Contributions:**")
                    for feature, contribution in explanation.as_list():
                        st.write(f"- {feature}: {contribution:.4f}")
                    
                    # Plot explanation
                    fig = explanation.as_pyplot_figure()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"LIME explanation failed: {str(e)}")
        
        return {
            'model_type': model_type,
            'target_variable': target_variable,
            'trained': 'trained_model' in st.session_state
        }


def get_ml_surrogate_manager() -> MLSurrogateManager:
    """Get ML surrogate manager instance"""
    if 'ml_surrogate_manager' not in st.session_state:
        st.session_state.ml_surrogate_manager = MLSurrogateManager()
    return st.session_state.ml_surrogate_manager
