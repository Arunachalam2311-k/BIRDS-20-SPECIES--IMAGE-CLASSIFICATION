# import libraries
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Page configuration and title
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: #336699;'>🕊️ BIRDS 20 SPECIES - IMAGE CLASSIFICATION 🕊️</h1><hr>", unsafe_allow_html=True)

# Load the trained model
MODEL_PATH = 'D:/work/my_cnn_model.h5'

try:
    model = load_model(MODEL_PATH)

    # Class names (must match the order of training)
    CLASS_NAMES = ['ABBOTTS BABBLER', 'ABBOTTS BOOBY', 'ABYSSINIAN GROUND HORNBILL', 'AFRICAN CROWNED CRANE',
                   'AFRICAN EMERALD CUCKOO', 'AFRICAN FIREFINCH', 'AFRICAN OYSTER CATCHER',
                   'AFRICAN PIED HORNBILL', 'AFRICAN PYGMY GOOSE', 'ALBATROSS', 'ALBERTS TOWHEE',
                   'ALEXANDRINE PARAKEET', 'ALPINE CHOUGH', 'ALTAMIRA YELLOWTHROAT', 'AMERICAN AVOCET',
                   'AMERICAN BITTERN', 'AMERICAN COOT', 'AMERICAN FLAMINGO', 'AMERICAN GOLDFINCH',
                   'AMERICAN KESTREL']
except Exception as e:
    st.error(f"❌ Error loading the model from {MODEL_PATH}: {e}")
    model = None
    CLASS_NAMES = []

# Initialize session states
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None
if 'predicted_image' not in st.session_state:
    st.session_state['predicted_image'] = None
if 'prediction_label' not in st.session_state:
    st.session_state['prediction_label'] = ""

# Layout with two columns
col1, col2 = st.columns([1, 1])

# Image Upload Section
with col1:
    st.markdown("### 📤 Upload Bird Image")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state['uploaded_image'] = image
        except Exception as e:
            st.error(f"Error loading image: {e}")

    if st.session_state['uploaded_image'] is not None:
        st.image(st.session_state['uploaded_image'], width=300, caption="Uploaded Image")
    else:
        st.markdown(
            """
            <div style='width: 100%; height: 300px; background-color: #f0f0f0;
                        display: flex; justify-content: center; align-items: center;
                        border: 2px dashed #ccc; color: #999;'>
                Drop image here to classify
            </div>
            """,
            unsafe_allow_html=True
        )

# Prediction Section
with col2:
    st.markdown("### 📊 Prediction Output")
    if st.session_state['uploaded_image'] is not None and model is not None:
        # Preprocess the image
        img = st.session_state['uploaded_image'].resize((150, 150))  # Same as model input
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        predicted_label = CLASS_NAMES[predicted_class]

        # Save to session
        st.session_state['predicted_image'] = st.session_state['uploaded_image']
        st.session_state['prediction_label'] = predicted_label

        # Display prediction result
        st.image(st.session_state['predicted_image'], width=300, caption=f"Predicted: {predicted_label}")

        st.success(f"🟢 Bird Identified: **{predicted_label}** (Confidence: {confidence:.2f})")
    else:
        st.markdown(
            """
            <div style='width: 100%; height: 300px; background-color: #f0f0f0;
                        display: flex; justify-content: center; align-items: center;
                        border: 2px dashed #ccc; color: #999;'>
                Prediction result will appear here
            </div>
            """,
            unsafe_allow_html=True
        )
        st.info("No prediction yet. Upload an image to get started.")


