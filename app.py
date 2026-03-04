import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

st.set_page_config(
    page_title="Specialty Coffee Predictor",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

dark_coffee_theme_css = """
<style>
    .stApp {
        background-color: #2b1d14;
    }
    
    [data-testid="stSidebar"] {
        background-color: #3e2723;
        border-right: 2px solid #5d4037;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, label, div, li {
        color: #EFEBE9 !important;
    }
    
    .stButton>button {
        background-color: #A16A4A !important; 
        color: #FFFFFF !important;
        border-radius: 8px;
        border: 1px solid #C49A7B;
        padding: 10px 24px;
        transition: all 0.3s ease;
        width: 100%;
        font-weight: bold;
        font-size: 16px;
    }
    
    .stButton>button:hover {
        background-color: #C49A7B !important;
        color: #2b1d14 !important;
        transform: scale(1.02);
        border: 1px solid #FFFFFF;
    }
    
    [data-testid="stMetricValue"] {
        color: #D4A373 !important;
    }
    
    [data-testid="stExpander"] {
        background-color: #4e342e !important;
        border: 1px solid #5d4037 !important;
    }
    
    hr {
        border-color: #5d4037 !important;
    }
    
    .stAlert {
        background-color: #4e342e !important;
        border: 1px solid #A16A4A !important;
    }
</style>
"""
st.markdown(dark_coffee_theme_css, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_columns():
    model_path = 'coffee_model.pkl'
    columns_path = 'model_columns.pkl'
    
    if not os.path.exists(model_path):
        return None, None
        
    model = joblib.load(model_path)
    try:
        model_columns = joblib.load(columns_path)
    except FileNotFoundError:
        model_columns = None
        
    return model, model_columns

model, expected_columns = load_model_and_columns()

st.title("☕ Specialty Coffee Quality Classifier")
st.markdown("**Predict the quality class of your coffee based on CQI sensory evaluations.**")
st.markdown("---")

st.sidebar.image("https://images.unsplash.com/photo-1511920170033-f8396924c348?w=500&auto=format&fit=crop&q=60", use_column_width=True)
st.sidebar.title("☕ Profile Inputs")

with st.sidebar.expander("Key Sensory Attributes", expanded=True):
    aroma = st.slider("Aroma", 0.0, 10.0, 7.5, 0.25)
    flavor = st.slider("Flavor", 0.0, 10.0, 7.5, 0.25)
    aftertaste = st.slider("Aftertaste", 0.0, 10.0, 7.5, 0.25)
    balance = st.slider("Balance", 0.0, 10.0, 7.5, 0.25)

with st.sidebar.expander("Taste & Mouthfeel", expanded=False):
    acidity = st.slider("Acidity", 0.0, 10.0, 7.5, 0.25)
    body = st.slider("Body", 0.0, 10.0, 7.5, 0.25)

with st.sidebar.expander("Physical Properties", expanded=False):
    moisture = st.number_input("Moisture", 0.0, 1.0, 0.10, 0.01)
    altitude = st.number_input("Altitude Mean (meters)", 0.0, 5000.0, 1200.0, 100.0)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analyze Current Profile")
    st.write("Click the button below to run the Random Forest model on the provided attributes.")
    
    if st.button("Predict Quality Class"):
        if model is None:
            st.error("Model file ('coffee_model.pkl') not found. Please generate it from your notebook.")
        else:
            with st.spinner('Brewing predictions...'):
                time.sleep(1.5) 
                
                input_dict = {col: 0 for col in (expected_columns if expected_columns else [])}
                input_dict.update({
                    'Aroma': aroma, 'Flavor': flavor, 'Aftertaste': aftertaste,
                    'Acidity': acidity, 'Body': body, 'Balance': balance,
                    'Moisture': moisture, 'altitude_mean_meters': altitude
                })
                
                input_df = pd.DataFrame([input_dict])
                if expected_columns is not None:
                    input_df = input_df[expected_columns]
                
                prediction = model.predict(input_df)[0]
                
                st.success("Analysis Complete!")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="Predicted Class", value=prediction)
                    
                with res_col2:
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_df)[0]
                        max_prob = np.max(probabilities) * 100
                        st.metric(label="Confidence Score", value=f"{max_prob:.1f}%")

with col2:
    st.info("Specialty coffee is graded on a 100-point scale. The attributes on the left are scored from 0-10 by certified Q-Graders. A higher balance between flavor, acidity, and body usually yields a Premium class.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #D4A373;'>Developed for Coffee Quality Classification | Powered by Machine Learning</p>", unsafe_allow_html=True)