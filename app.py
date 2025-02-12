import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from fpdf import FPDF

st.set_page_config(page_title="Credit Default Risk", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CA5AF;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #065159;
    }
    .stHeader {
        color: #4CA5AF;
        font-size: 24px;
        font-weight: bold;
    }
    .stSubheader {
        color: #333;
        font-size: 20px;
        font-weight: bold;
    }
    .stMarkdown {
        color: #999;
        font-size: 16px;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stSidebar {
        background-color: #4CA5AF;
        color: white;
        padding: 10px;
        border-radius: 10px;
        width: 250px;
    }
    .stSidebar .stMarkdown {
        color: white;
    }
    .stSidebar .stButton {
        background-color: #4CA5AF;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stSidebar .stButton>button:hover {
        background-color: #065159;
    }
    .stSidebar .stNumberInput>div>div>input {
        background-color: black;
        color: white;
    }
    .stSidebar .stSelectbox>div>div>div {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_components():
    return (
        joblib.load('lgbm_model.joblib'),
        joblib.load('selected_features.joblib'),
        joblib.load('feature_values.joblib'),
        joblib.load('categorical_features.joblib')
    )

model, selected_features, feature_values, cat_features = load_components()

# App title with HTML
st.markdown(
    """
    <div class="main">
        <h1 style="color: #4CA5AF; text-align: center;">🏦 Home Credit Default Risk Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div>
        <h3></h3>
        <h2 style="color: #4CA5AF;">Welcome!</h2>
        <h3>To veiw your prediction analysis please fill the details on the left!</h3>
        <h4> __________________________________________________________________</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

# Layout for user input
st.sidebar.markdown(
    """
    <div class="stSidebar">
        <h2 style="color: white; text-align: left;">📊 Enter Customer Data</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

input_data = {}
selected_features = list(set(selected_features))
for feature in selected_features:
    if feature in cat_features:
        options = feature_values.get(feature, [])
        input_data[feature] = st.sidebar.selectbox(f"📌 {feature}", options)
    else:
        bounds = feature_values.get(feature, {'min': 0, 'max': 1})
        min_val = max(bounds.get('min', 0), -1e6)
        max_val = min(bounds.get('max', 1), 1e6)
        default = (min_val + max_val) / 2
        input_data[feature] = st.sidebar.number_input(
            f"📌 {feature}", 
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default)
        )

input_df = pd.DataFrame([input_data])

# Preprocessing function
def preprocess(df):
    for col in cat_features:
        if col in df.columns:
            le = LabelEncoder()
            le.classes_ = np.array(feature_values[col])
            df[col] = le.transform(df[col].astype(str))
    return df

# Predict button
if st.sidebar.button("🔍 Predict"):
    st.markdown(f"""
    <div>
        <h2 style="color: white; text-align: centre">Prediction Analysis:</h2>
    </div>
    """,
    unsafe_allow_html=True,
    )

    processed_df = preprocess(input_df)
    proba = model.predict_proba(processed_df)[0][1]

    # Risk status and color coding
    if proba >= 0.5:
        status = "🔴 High Risk"
        color = "red"
    else:
        status = "🟢 Low Risk"
        color = "green"
    
    # Layout for results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(
            f"""
            <div>
                <h2 style="color: {color};">{status}</h2>
                <h3>Probability: {proba:.2%}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed_df)
    
    # Feature importance
    st.subheader("📊 Top Influencing Features")
    impacts = pd.DataFrame({
        'Feature': selected_features,
        'Impact': shap_values[0]
    }).sort_values('Impact', key=abs, ascending=False).head(5)

    # Color based on impact direction
    impacts["Color"] = impacts["Impact"].apply(lambda x: "red" if x > 0 else "green")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(y=impacts["Feature"], x=impacts["Impact"], palette=impacts["Color"].tolist(), ax=ax)
    ax.set_title("Top Influencing Factors")
    st.pyplot(fig)

    # Interactive SHAP Waterfall plot
    st.subheader("🔎 Detailed Feature Contribution")

    # Create a figure for SHAP waterfall plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Generate SHAP waterfall plot
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value,  # Use correct expected value for positive class
            data=processed_df.iloc[0]
        )
    )

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Add explanation for better interpretability
    st.markdown("""
    ### Understanding the Waterfall Plot:
    - **X-Axis**: Contribution of each feature to the final prediction (log-odds space).
    - **Y-Axis**: Features affecting the decision, ranked from highest impact to lowest.
    - **Base Value**: The model’s average predicted log-odds before considering individual features.
    - **Feature Values**: Numbers beside each feature indicate their actual input value.
    - **Positive (Red) Bars**: Features that **increase the default risk**.
    - **Negative (Blue) Bars**: Features that **decrease the default risk**.
    """)

    st.subheader("🔎 Feature Contribution of all features")

    st.markdown("Hover over the bars to see exact SHAP values and feature contributions.")

    # Create Plotly waterfall plot
    waterfall_data = {
        "Feature": ["Base Value"] + selected_features,
        "SHAP Value": [explainer.expected_value] + list(shap_values[0]),
        "Color": ["gray"] + ["red" if x > 0 else "green" for x in shap_values[0]]
    }

    fig = go.Figure(go.Waterfall(
        name="Feature Contributions",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(selected_features),
        x=waterfall_data["Feature"],
        y=waterfall_data["SHAP Value"],
        text=[f"{val:.4f}" for val in waterfall_data["SHAP Value"]],
        textposition="outside",
        decreasing={"marker": {"color": "green"}},
        increasing={"marker": {"color": "red"}},
        totals={"marker": {"color": "gray"}},
        connector={"line": {"color": "gray"}},
    ))

    fig.update_layout(
        title="Feature Contributions to Default Risk",
        xaxis_title="Features",
        yaxis_title="SHAP Value (Impact on Prediction)",
        showlegend=False,
        height=800,
        margin=dict(l=100, r=50, t=80, b=50),
        xaxis_tickangle=-45,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Understanding this chart:
    - **Red Bars**: Features that increase the risk of default.
    - **Green Bars**: Features that decrease the risk of default.
    - **Hover over bars** to see exact SHAP impact values.
    - **X-Axis**: Top influencing features sorted by importance.
    - **Y-Axis**: How much each feature pushes the risk prediction up or down.
    """)

    # Feature Value Comparison
    st.subheader("📈 Feature Value Comparison")
    st.markdown("Compare the input feature values with the dataset's average.")

    comparison_data = []
    for feature in selected_features:
        if feature in cat_features:
            comparison_data.append({
                "Feature": feature,
                "Input Value": input_data[feature],
                "Dataset Average": feature_values[feature][0]  # Use first category as average
            })
        else:
            comparison_data.append({
                "Feature": feature,
                "Input Value": input_data[feature],
                "Dataset Average": np.median(list(feature_values[feature].values()))
            })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df)

    # Downloadable Report
    st.subheader("📄 Download Report")
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Home Credit Default Risk Prediction Report", ln=True, align="C")
        pdf.cell(200, 10, txt=f"Risk Status: {status}", ln=True, align="L")
        pdf.cell(200, 10, txt=f"Probability: {proba:.2%}", ln=True, align="L")
        pdf.cell(200, 10, txt="Top Influencing Features:", ln=True, align="L")
        for _, row in impacts.iterrows():
            pdf.cell(200, 10, txt=f"- {row['Feature']}: {row['Impact']:.4f}", ln=True, align="L")
        pdf.output("report.pdf")
        st.success("Report generated successfully!")
        with open("report.pdf", "rb") as file:
            st.download_button("Download Report", file, file_name="report.pdf")

