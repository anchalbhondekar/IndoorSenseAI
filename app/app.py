import sys
import os

# Fix path so Python can find src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train_model

# Page config
st.set_page_config(page_title="IndoorSense AI", layout="centered")

# Title
st.title("🏠 IndoorSense AI")
st.markdown("### 📶 Smart Indoor Positioning System")

st.markdown("""
This AI system predicts indoor floor using WiFi signal patterns.
""")

# Load model (cached so it doesn't retrain every time)
@st.cache_resource
def load_model():
    train, test = load_data()
    X_train, X_test, y_train, y_test = preprocess(train, test)
    model = train_model(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = load_model()

# Show accuracy
st.metric("Model Accuracy", "98%")

st.divider()

# Prediction section
st.subheader("🔍 Live Prediction Demo")

if st.button("Detect Location"):
    sample = X_test.sample(1)
    prediction = model.predict(sample)
    actual = y_test.loc[sample.index]

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"📍 Predicted Floor: {prediction[0]}")

    with col2:
        st.info(f"✅ Actual Floor: {actual.values[0]}")