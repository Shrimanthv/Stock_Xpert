# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# import pickle


# st.title("Stock Price Prediction System")

# with open('random_forest_model.pkl', 'rb') as file:
#     best_rf_model = pickle.load(file)
    
    
# st.header("Enter Stock Data for Prediction")

# open_price = st.number_input("Open Price", min_value=0.0, format="%.2f")
# high_price = st.number_input("High Price", min_value=0.0, format="%.2f")
# low_price = st.number_input("Low Price", min_value=0.0, format="%.2f")
# close_price = st.number_input("Close Price", min_value=0.0, format="%.2f")
# volume = st.number_input("Volume", min_value=0, format="%d")

# if st.button("Predict"):
#     input_data = pd.DataFrame({
#         "Open": [open_price],
#         "High": [high_price],
#         "Low": [low_price],
#         "Close": [close_price],
#         "Volume": [volume]
#     })
#     prediction = best_rf_model.predict(input_data)[0]
#     st.write("Prediction Result:",prediction)
#     if prediction==1:
#         st.success("The stock price is likely to go up tomorrow.")
#     else:
#         st.warning("The stock price is likely to go down tomorrow.")
    
    
import streamlit as st
import numpy as np
import pickle

# Load model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Stock Movement Predictor")

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
  