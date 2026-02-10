# =====================================================================
# MAIN APPLICATION - Football Analytics Hub
# Multi-page Streamlit Application
# =====================================================================

import streamlit as st

# =====================================================================
# Page Configuration
# =====================================================================
st.set_page_config(
    page_title="⚽ Football Analytics Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# Main Landing Page
# =====================================================================

st.title("⚽ Football Analytics Hub")
st.markdown("---")

st.markdown("""
## Welcome to the Football Analytics Platform

This comprehensive platform provides two powerful tools for football analytics:

### 📊 Available Tools:

#### 1. **RAG Q&A System** 
- Ask natural language questions about player performance
- Get AI-powered insights based on historical data
- Filter by player, season, and club
- Powered by FAISS vector database and Hugging Face LLM

#### 2. **Performance Prediction** 
- Upload player statistics and get performance predictions
- ML-powered forecasting using Ridge Regression
- Trained on 2019-2024 data from top European leagues
- Instant CSV export of predictions

### 🚀 Getting Started

Use the **sidebar navigation** (☰) to select a tool and start your analysis!

""")

# =====================================================================
# Sidebar Information
# =====================================================================
st.sidebar.title("Navigation")
st.sidebar.info("""
Select a page from above to access different analytics tools.

**Pages:**
- 🏠 Home (Current)
- 🤖 RAG Q&A System
- 📈 Performance Predictor
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About
This platform combines:
- Retrieval-Augmented Generation (RAG)
- Machine Learning Predictions
- Interactive Data Exploration

**Data Source:** FBREF Top 7 European Leagues (2019-2024)
""")

# =====================================================================
# Feature Highlights
# =====================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 RAG Q&A Features")
    st.markdown("""
    - **Natural Language Queries**: Ask questions in plain English
    - **Semantic Search**: Find relevant players using FAISS
    - **AI Analysis**: Get intelligent insights from DeepSeek LLM
    - **Advanced Filters**: Filter by player, season, club
    - **Context Display**: See the data behind each answer
    """)

with col2:
    st.subheader("📈 Prediction Features")
    st.markdown("""
    - **Batch Predictions**: Upload CSV with multiple players
    - **Accurate Forecasting**: Ridge Regression model
    - **Data Validation**: Automatic column checking
    - **Missing Value Handling**: Smart data imputation
    - **Easy Export**: Download predictions as CSV
    """)

st.markdown("---")

# =====================================================================
# Quick Stats
# =====================================================================
st.subheader("📊 Platform Statistics")

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.metric("Data Coverage", "2019-2024")
    
with stat_col2:
    st.metric("Leagues Covered", "7 Top European")
    
with stat_col3:
    st.metric("Models Available", "2 Tools")
    
with stat_col4:
    st.metric("ML Accuracy", "R² > 0.85")

st.markdown("---")

# =====================================================================
# Footer
# =====================================================================
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: #888;'>
        Built with Streamlit | Data from FBREF | Powered by FAISS & Hugging Face
    </p>
</div>
""", unsafe_allow_html=True)