import streamlit as st
import tensorflow as tf
import numpy as np

# Load trained model (assumes model trained on Iris dataset)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("iris_model.keras")

model = load_model()

st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸")

st.title("🌸 Iris Flower Predictor")
st.write("Enter the measurements of the iris flower below to predict its species:")

# Input fields for Iris features
st.subheader("Flower Measurements (in cm)")
user_input = []
cols = st.columns(2)  # Two columns for inputs
feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
for i, name in enumerate(feature_names):
    with cols[i % 2]:
        value = st.number_input(f"{name}", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        user_input.append(value)

# Prediction button
if st.button("🔮 Predict Flower Species"):
    input_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_data, verbose=0)[0]
    
    # Class names for Iris dataset
    class_names = ["Setosa", "Versicolor", "Virginica"]
    predicted_class = np.argmax(prediction)
    predicted_prob = prediction[predicted_class]

    st.subheader("📌 Prediction Result")
    st.success(f"Predicted Species: **{class_names[predicted_class]}** (Probability: {predicted_prob:.4f})")
    
    # Display probabilities for all classes
    st.write("Class Probabilities:")
    for name, prob in zip(class_names, prediction):
        st.markdown(f"- {name}: {prob:.4f}")
