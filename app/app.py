import streamlit as st
import pandas as pd

from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train_model

st.set_page_config(page_title="IndoorSense AI", layout="centered")

st.title("🏠 IndoorSense AI")
st.markdown("### 📶 Indoor Position Detection System")

st.write("This system predicts indoor floor using WiFi signal patterns")

# Load model
@st.cache_resource
def load_model():
    train, test = load_data()
    X_train, X_test, y_train, y_test = preprocess(train, test)
    model = train_model(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = load_model()

# Show sample data
st.subheader("📊 Sample Data")
if st.checkbox("Show dataset sample"):
    st.write(X_test.head())

# Predict button
st.subheader("🔍 Test Prediction")

if st.button("Predict Random Location"):
    
    sample = X_test.sample(1)
    prediction = model.predict(sample)
    actual = y_test.loc[sample.index]

    st.success(f"📍 Predicted Floor: {prediction[0]}")
    st.info(f"✅ Actual Floor: {actual.values[0]}")