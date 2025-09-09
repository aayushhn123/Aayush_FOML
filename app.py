# app.py
import streamlit as st
import tensorflow as tf
import numpy as np

# Load trained model (make sure you have run train.py first)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

st.set_page_config(page_title="TensorFlow + Streamlit", page_icon="🤖")

st.title("🤖 TensorFlow Model Deployment with Streamlit")
st.write("Enter 10 numeric features below to get a prediction:")

# Input fields
user_input = []
cols = st.columns(5)  # two rows of inputs
for i in range(10):
    with cols[i % 5]:
        value = st.number_input(f"Feature {i+1}", value=0.0)
        user_input.append(value)

# Prediction button
if st.button("🔮 Predict"):
    input_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_data, verbose=0)[0][0]

    st.subheader("📌 Prediction Result")
    st.success(f"Prediction Score: **{prediction:.4f}**")

    if prediction > 0.5:
        st.markdown("✅ The model predicts: **Class 1**")
    else:
        st.markdown("❌ The model predicts: **Class 0**")
