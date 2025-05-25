import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Constants
MODEL_PATH = "oral_cancer_detection_model.h5"
FILE_ID = "1kk-QC6j4zbEhVJ6C1zUHVtXPx_2paWWa"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if not already present
@st.cache_resource
def load_cancer_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_cancer_model()

# Label mapping
label_map = {0: 'Non-Cancer', 1: 'Cancer'}

# Streamlit UI
st.title("🧪 Oral Cancer Detection App")
st.write("Upload an image to classify it as **Cancer** or **Non-Cancer** using a deep learning model.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        # Preprocess image
        img = image.resize((224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0][0]
        pred_label = 1 if prediction > 0.5 else 0
        confidence = prediction if pred_label == 1 else 1 - prediction

        # Display result
        st.success(f"Prediction: **{label_map[pred_label]}**")
        st.info(f"Confidence: **{confidence * 100:.2f}%**")
