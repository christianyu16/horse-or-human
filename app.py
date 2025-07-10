import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("horse_or_human_model.h5")

model = load_model()

# App title
st.title("🐴 Horse or Human Classifier")
st.write("Upload an image and I'll tell you if it's a **horse** or a **human**!")

# File uploader
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Display uploaded image
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((150, 150))  # Resize to match training
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normalize and batch

    # Prediction
    prediction = model.predict(img_array)[0][0]
    label = "Human 🧍" if prediction > 0.5 else "Horse 🐴"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Show prediction
    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"Confidence: `{confidence:.2%}`")
