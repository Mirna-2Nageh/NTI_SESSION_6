import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ========= CONFIG =========
MODEL_PATH = "flowers_cnn.h5"   # use the same file you saved in training
IMG_SIZE = (128, 128)           # must match training size
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

# ========= LOAD MODEL =========
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ========= PREPROCESS =========
def preprocess(img: Image.Image):
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    if arr.shape[-1] == 4:  # remove alpha channel if exists
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

# ========= STREAMLIT UI =========
st.title("🌸 Flowers Recognition App")
st.write("Upload a flower image, and the CNN model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess(image)
    preds = model.predict(img_array, verbose=0)[0]
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    st.subheader(f"Prediction: **{pred_class}**")
    st.write(f"Confidence: **{confidence:.3f}**")

    st.bar_chart({CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))})
