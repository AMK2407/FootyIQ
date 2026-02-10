# =====================================================================
# PAGE 2: PERFORMANCE PREDICTOR
# ML-based Football Player Performance Prediction
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os

# =====================================================================
# Page Configuration
# =====================================================================
st.set_page_config(
    page_title="⚽ Performance Predictor",
    page_icon="📈",
    layout="wide"
)

# =====================================================================
# Lazy Imports (Only load when this page is accessed)
# =====================================================================
@st.cache_resource
def lazy_import_ml_dependencies():
    """Import ML-specific dependencies only when needed."""
    try:
        import joblib
        return {'joblib': joblib}
    except ImportError as e:
        st.error(f"""
        ❌ ML dependencies not found: {e}
        
        Please install required packages:
        ```
        pip install joblib scikit-learn
        ```
        """)
        return None

# =====================================================================
# Configuration
# =====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    MODEL_FILE = os.path.join(MODEL_DIR, "ridge_model.pkl")
    SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
    FEATURES_FILE = os.path.join(MODEL_DIR, "features.pkl")

# =====================================================================
# Load Model Artifacts
# =====================================================================
@st.cache_resource
def load_artifacts():
    """Load model, scaler, and feature list."""
    deps = lazy_import_ml_dependencies()
    if deps is None:
        st.stop()
    
    joblib = deps['joblib']
    
    try:
        model = joblib.load(Config.MODEL_FILE)
        scaler = joblib.load(Config.SCALER_FILE)
        features = joblib.load(Config.FEATURES_FILE)
        
        st.success(f"✅ Model loaded successfully with {len(features)} features")
        return model, scaler, features
    
    except FileNotFoundError as e:
        st.error(f"""
        ❌ Model files not found!
        
        Expected files:
        - {Config.MODEL_FILE}
        - {Config.SCALER_FILE}
        - {Config.FEATURES_FILE}
        
        Please ensure the model directory exists and contains the trained model artifacts.
        """)
        st.stop()
    
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        st.stop()

# =====================================================================
# Helper Functions
# =====================================================================
def validate_dataframe(df: pd.DataFrame, required_features: list) -> tuple:
    """
    Validate uploaded dataframe against required features.
    
    Returns:
        tuple: (is_valid, missing_columns, extra_columns)
    """
    missing_cols = list(set(required_features) - set(df.columns))
    extra_cols = list(set(df.columns) - set(required_features))
    is_valid = len(missing_cols) == 0
    
    return is_valid, missing_cols, extra_cols

def prepare_features(df: pd.DataFrame, required_features: list) -> pd.DataFrame:
    """
    Prepare features for prediction.
    
    - Select required columns
    - Convert to numeric
    - Handle missing values
    """
    X = df[required_features].copy()
    
    # Convert to numeric, coercing errors to NaN
    X = X.apply(pd.to_numeric, errors="coerce")
    
    # Check for missing values
    missing_count = X.isnull().sum().sum()
    
    if missing_count > 0:
        st.warning(f"⚠️ Found {missing_count} missing or invalid values. Filling with column mean.")
        X = X.fillna(X.mean())
    
    return X

def display_prediction_summary(df: pd.DataFrame, predictions: np.ndarray):
    """Display summary statistics of predictions."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(predictions))
    
    with col2:
        st.metric("Mean Prediction", f"{predictions.mean():.2f}")
    
    with col3:
        st.metric("Max Prediction", f"{predictions.max():.2f}")
    
    with col4:
        st.metric("Min Prediction", f"{predictions.min():.2f}")

# =====================================================================
# Main App
# =====================================================================
def main():
    st.title("📈 Football Player Performance Predictor")
    st.markdown("""
    Upload a **CSV file** containing player statistics to predict performance metrics.
    
    The model uses **Ridge Regression** trained on data from 2019-2022 and tested on 2023-2024.
    """)
    
    # Load Model
    model, scaler, FEATURES = load_artifacts()
    
    # Sidebar Info
    st.sidebar.header("ℹ️ Model Information")
    st.sidebar.write(f"**Model Type:** Ridge Regression")
    st.sidebar.write(f"**Training Period:** 2019-2022")
    st.sidebar.write(f"**Test Period:** 2023-2024")
    st.sidebar.write(f"**Features Used:** {len(FEATURES)}")
    
    with st.sidebar.expander("📋 View Required Features"):
        st.write(FEATURES)
    
    st.sidebar.markdown("---")
    
    # Download Sample Template
    st.sidebar.subheader("📥 Download Template")
    if st.sidebar.button("Generate Sample CSV Template"):
        # Create a sample dataframe with required columns
        sample_df = pd.DataFrame(columns=FEATURES)
        sample_csv = sample_df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="⬇️ Download Template CSV",
            data=sample_csv,
            file_name="football_prediction_template.csv",
            mime="text/csv"
        )
    
    # Main Content
    st.markdown("---")
    
    # File Upload
    st.subheader("📂 Upload Player Statistics")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file containing player statistics with the required features"
    )
    
    if uploaded_file is not None:
        
        # Read CSV
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")
            return
        
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.info(f"📊 Dataset contains **{len(df)}** rows and **{len(df.columns)}** columns")
        
        # Validate Columns
        is_valid, missing_cols, extra_cols = validate_dataframe(df, FEATURES)
        
        if not is_valid:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.info("💡 Please ensure your CSV contains all required features. You can download a template from the sidebar.")
            return
        
        if extra_cols:
            st.info(f"ℹ️ Extra columns found (will be ignored): {extra_cols}")
        
        # Prepare Data
        st.subheader("⚙️ Preparing Data for Prediction")
        
        with st.spinner("Processing features..."):
            X = prepare_features(df, FEATURES)
        
        st.success("✅ Features prepared successfully")
        
        # Show data quality info
        with st.expander("🔍 View Data Quality Report"):
            st.write("**Feature Statistics:**")
            st.dataframe(X.describe())
        
        # Prediction Button
        st.markdown("---")
        
        if st.button("🔮 Generate Predictions", type="primary"):
            
            # Make Predictions
            with st.spinner("🤖 Running prediction model..."):
                try:
                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)
                    predictions_rounded = np.round(predictions, 2)
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    return
            
            st.success("✅ Predictions generated successfully!")
            
            # Add predictions to dataframe
            df["Predicted_Performance"] = predictions_rounded
            
            # Display Results
            st.subheader("📊 Prediction Results")
            
            # Summary Statistics
            display_prediction_summary(df, predictions)
            
            st.markdown("---")
            
            # Full Results Table
            st.write("**Full Results:**")
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Prediction Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Histogram of Predictions**")
                st.bar_chart(pd.DataFrame(predictions, columns=['Predictions']))
            
            with col2:
                st.write("**Top 10 Predictions**")
                top_10 = df.nlargest(10, 'Predicted_Performance')[
                    ['Predicted_Performance']
                ].head(10)
                st.dataframe(top_10)
            
            # Download Results
            st.markdown("---")
            st.subheader("💾 Download Results")
            
            csv_data = df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv_data,
                file_name="football_predictions.csv",
                mime="text/csv"
            )
    
    else:
        # No file uploaded
        st.info("👆 Please upload a CSV file to start making predictions.")
        
        st.markdown("---")
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Prepare your data**: Ensure your CSV contains all required features
        2. **Download template** (optional): Use the sidebar to get a template CSV
        3. **Upload CSV**: Click the upload button above
        4. **Review data**: Check the preview and validation results
        5. **Generate predictions**: Click the prediction button
        6. **Download results**: Export predictions as CSV
        """)
        
        st.markdown("---")
        st.subheader("💡 Tips")
        st.markdown("""
        - Ensure all numerical columns are properly formatted
        - Missing values will be automatically filled with column means
        - Extra columns in your CSV will be ignored
        - Check the required features list in the sidebar
        """)

if __name__ == '__main__':
    main()