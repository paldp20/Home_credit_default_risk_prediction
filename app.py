import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import LabelEncoder


@st.cache_resource
def load_components():
    return (
        joblib.load('lgbm_model.joblib'),
        joblib.load('selected_features.joblib'),
        joblib.load('feature_values.joblib'),
        joblib.load('categorical_features.joblib')
    )

model, selected_features, feature_values, cat_features = load_components()

st.title("Home Credit Default Risk Prediction")

# Collect only the 143 selected features
input_data = {}
for feature in selected_features:
    if feature in cat_features:
        options = feature_values.get(feature, [])
        input_data[feature] = st.selectbox(feature, options)
    else:
        bounds = feature_values.get(feature, {'min': 0, 'max': 1})
        min_val = max(bounds.get('min', 0), -1e6)
        max_val = min(bounds.get('max', 1), 1e6)
        default = (min_val + max_val) / 2
        input_data[feature] = st.number_input(
            feature, 
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default)
        )

# Create DataFrame with ONLY the 143 features
input_df = pd.DataFrame([input_data])

# Preprocessing
def preprocess(df):
    # Label encode categoricals
    for col in cat_features:
        if col in df.columns:
            le = LabelEncoder()
            le.classes_ = np.array(feature_values[col])
            df[col] = le.transform(df[col].astype(str))
    return df

if st.button("Predict"):
    processed_df = preprocess(input_df)
    proba = model.predict_proba(processed_df)[0][1]
    
    # Display results
    status = "🔴 High Risk" if proba >= 0.5 else "🟢 Low Risk"
    st.markdown(f"## {status} (Probability: {proba:.2%})")
    
    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed_df)
    
    st.subheader("Top Influencing Factors")
    impacts = pd.DataFrame({
        'Feature': selected_features,
        'Impact': shap_values[0]
    }).sort_values('Impact', key=abs, ascending=False).head(5)
    
    for _, row in impacts.iterrows():
        direction = "increased" if row.Impact > 0 else "decreased"
        st.write(f"- **{row.Feature}** ({direction} risk)")