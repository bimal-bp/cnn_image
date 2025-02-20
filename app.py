import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Google Drive link for model
drive_link = "https://drive.google.com/uc?id=1os3m_b2PYcvCz33Ku_vzRjSkhBh8Y7e5"
model_path = "resnet_vit_model.h5"

# Download model if not exists
if not os.path.exists(model_path):
    gdown.download(drive_link, model_path, quiet=False)

# Load model
try:
    model = tf.keras.models.load_model(model_path, compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Define class labels
class_names = [
    "a_Good", "b_Moderate", "c_Unhealthy_for_Sensitive_Groups",
    "d_Unhealthy", "e_Very_Unhealthy", "f_Severe"
]

# Streamlit UI
st.title("Air Quality Classification")
st.write("Upload an image to classify air quality.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

def preprocess_image(image):
    """Preprocess the image."""
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image  # Shape: (1, 224, 224, 3)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify"):
        if model is None:
            st.error("Model is not loaded. Please check for errors.")
        else:
            try:
                # Load and preprocess image
                image = load_img(uploaded_file, target_size=(224, 224))
                image_preprocessed = preprocess_image(image)

                # Make prediction
                predictions = model.predict(image_preprocessed)
                predicted_class = class_names[np.argmax(predictions)]

                # Display results
                st.subheader(f"Prediction: {predicted_class}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
