
import streamlit as st
import numpy as np
import pickle

# Load model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📈  StockXpert — Stock Movement Prediction using Random Forest Classifier")

st.write("Enter the values for the features below:")

# User input for all features (same as used in training)
open_val = st.number_input("Open", min_value=0.0, value=1000.0)
high = st.number_input("High", min_value=0.0, value=1050.0)
low = st.number_input("Low", min_value=0.0, value=980.0)
close = st.number_input("Close", min_value=0.0, value=1020.0)
volume = st.number_input("Volume", min_value=0.0, value=3000000.0)

if st.button("Predict"):
    # Prepare data for prediction
    input_data = np.array([[open_val, high, low, close, volume]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Target: {prediction}")
    if prediction==1:
        st.success("The stock price is likely to go up tomorrow.")
    else:
        st.warning("The stock price is likely to go down tomorrow.")
  