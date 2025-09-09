# app.py
import streamlit as st
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("my_model.keras")

st.title("🔮 Simple TensorFlow Model Deployment with Streamlit")
st.write("Enter 10 numeric features to get a prediction:")

# Input fields
user_input = []
for i in range(10):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    user_input.append(value)

# Predict button
if st.button("Predict"):
    input_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_data)[0][0]

    st.success(f"Prediction Score: {prediction:.4f}")

    if prediction > 0.5:
        st.write("✅ The model predicts: **Class 1**")
    else:
        st.write("❌ The model predicts: **Class 0**")
