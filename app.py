import streamlit as st
import os
import gdown
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "horse_or_human_model.h5"
FILE_ID = "1_du-Dk7x_cM9xUJs9icHeoG7bdTFOs7X"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

st.title("🐴 Horse or Human Classifier")
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    img = image.resize((150, 150))
    x = np.expand_dims(np.array(img) / 255.0, 0)
    pred = model.predict(x)[0][0]

    label = "Human" if pred > 0.5 else "Horse"
    conf = pred if pred > 0.5 else 1 - pred
    st.write(f"**{label}** with {conf:.2%} confidence")
