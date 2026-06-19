import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/iris_model.joblib")

@st.cache_resource
def load_model_info():
    with open("models/model_info.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_feature_ranges():
    with open("models/feature_ranges.json", "r") as f:
        return json.load(f)

model = load_model()
model_info = load_model_info()
feature_ranges = load_feature_ranges()

st.title("🌸 Iris Flower Classification")

st.write(
    "Predict the species of an Iris flower using a trained Random Forest model."
)

col1, col2 = st.columns([2, 1])

with col1:

    sepal_length = st.slider(
        "Sepal Length (cm)",
        float(feature_ranges["sepal_length"]["min"]),
        float(feature_ranges["sepal_length"]["max"]),
        float(feature_ranges["sepal_length"]["default"])
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        float(feature_ranges["sepal_width"]["min"]),
        float(feature_ranges["sepal_width"]["max"]),
        float(feature_ranges["sepal_width"]["default"])
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        float(feature_ranges["petal_length"]["min"]),
        float(feature_ranges["petal_length"]["max"]),
        float(feature_ranges["petal_length"]["default"])
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        float(feature_ranges["petal_width"]["min"]),
        float(feature_ranges["petal_width"]["max"]),
        float(feature_ranges["petal_width"]["default"])
    )

with col2:

    st.subheader("Current Values")

    df = pd.DataFrame({
        "Feature": [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ],
        "Value": [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    })

    st.dataframe(df, hide_index=True)

if st.button("Predict Species"):

    features = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    species = model_info["target_names"][prediction]

    st.success(f"Predicted Species: {species}")

    st.subheader("Confidence Scores")

    for i, prob in enumerate(probabilities):
        st.write(
            f"{model_info['target_names'][i]} : {prob*100:.2f}%"
        )
        st.progress(float(prob))

st.markdown("---")
st.write("Built with Streamlit and Scikit-Learn")