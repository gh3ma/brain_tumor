import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_cnn_model.keras")

model = load_model()

st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image and the model will predict whether a tumor is present.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction Result:")
    
    st.write("Raw prediction value:", round(float(prediction), 4))

    if prediction >= 0.5:
        st.error("🧠 Tumor Detected")
        confidence = prediction * 100
    else:
        st.success("✅ No Tumor Detected")
        confidence = (1 - prediction) * 100

    st.write(f"Confidence: {confidence:.2f}%")
