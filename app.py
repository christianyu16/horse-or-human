import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("https://huggingface.co/zero-gravity-ai/horse-or-human/resolve/main/horse_or_human_model.h5")

st.title("Horse or Human?")
uploaded = st.file_uploader("Upload an image:", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)
    x = img.resize((150,150))
    x = np.expand_dims(np.array(x)/255.0,0)
    pred = model.predict(x)[0][0]
    label = "Human" if pred>0.5 else "Horse"
    conf = pred if pred>0.5 else 1-pred
    st.write(f"**{label}** with {conf:.2%} confidence")
