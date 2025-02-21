import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Google Drive file ID (Extracted from the link)
drive_file_id = "1MGBH4qECimwgJGXLuEv2Y_ZEUV9b0Yql"
model_path = "resnet_vit_model.h5"

# Function to download model
def download_model():
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... Please wait."):
            gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", model_path, quiet=False)

# Download model if not exists
download_model()

# Load model
try:
    with st.spinner("Loading model..."):
        model = tf.keras.models.load_model(model_path, compile=False)
    st.success("Model successfully loaded!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Define class labels and air quality descriptions
class_names = [
    "a_Good", "b_Moderate", "c_Unhealthy_for_Sensitive_Groups",
    "d_Unhealthy", "e_Very_Unhealthy", "f_Severe"
]

aq_descriptions = {
    "a_Good": {"Range": "0-50", "Label": "Good", "Description": "Air quality is satisfactory, and air pollution poses little or no risk."},
    "b_Moderate": {"Range": "51-100", "Label": "Moderate", "Description": "Air quality is acceptable; however, some pollutants may be a concern for very sensitive individuals."},
    "c_Unhealthy_for_Sensitive_Groups": {"Range": "101-150", "Label": "Unhealthy for Sensitive Groups", "Description": "Children, elderly, and those with respiratory conditions should limit outdoor activities."},
    "d_Unhealthy": {"Range": "151-200", "Label": "Unhealthy", "Description": "Everyone may begin to experience health effects; sensitive groups should avoid outdoor exertion."},
    "e_Very_Unhealthy": {"Range": "201-300", "Label": "Very Unhealthy", "Description": "Health warnings of emergency conditions. The entire population is more likely to be affected."},
    "f_Severe": {"Range": "301-500", "Label": "Hazardous", "Description": "Serious health effects; everyone should avoid outdoor activities."}
}

# Streamlit UI
st.title("🌍 Air Quality Classification")
st.write("Upload an image to classify the air quality level.")

uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "png"])

# Image preprocessing function
def preprocess_image(image):
    """Preprocess the image for prediction."""
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image  # Shape: (1, 224, 224, 3)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Classify"):
        if model is None:
            st.error("⚠️ Model is not loaded. Please check for errors.")
        else:
            try:
                # Load and preprocess image
                image = load_img(uploaded_file, target_size=(224, 224))
                image_preprocessed = preprocess_image(image)

                # Make prediction
                predictions = model.predict(image_preprocessed)
                predicted_index = np.argmax(predictions)
                predicted_class = class_names[predicted_index]


                # Get AQI details
                result = aq_descriptions[predicted_class]

                # Display results
                st.subheader(f"🌡 Prediction: {result['Label']} ({result['Range']})")

                st.write(result['Description'])

                # Show warning for hazardous air quality
                if predicted_class in ["e_Very_Unhealthy", "f_Severe"]:
                    st.warning("🚨 High pollution levels detected! Limit outdoor exposure.")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
