import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Download the model from Google Drive if not available
drive_url = "https://drive.google.com/uc?id=1MGBH4qECimwgJGXLuEv2Y_ZEUV9b0Yql"
model_path = "mobilenetv2_model.h5"

if not os.path.exists(model_path):
    st.info("Downloading model... Please wait.")
    gdown.download(drive_url, model_path, quiet=False)

# Load the trained model
st.info("Loading model...")
model = tf.keras.models.load_model(model_path)
st.success("Model loaded successfully!")

# Define class labels
class_labels = [
    "a_Good",
    "b_Moderate",
    "c_Unhealthy_for_Sensitive_Groups",
    "d_Unhealthy",
    "e_Very_Unhealthy",
    "f_Severe"
]

# Function to preprocess image
def preprocess_image(image):
    """Preprocesses the image for model prediction."""
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🌍 Air Quality Prediction App 🏭")
st.write("Upload an image, and the AI model will classify its air quality.")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)
    
    if st.button("🚀 Predict"):
        try:
            # Load and preprocess image
            image = load_img(uploaded_file, target_size=(224, 224))
            image_processed = preprocess_image(image)

            # Predict
            predictions = model.predict(image_processed)
            predicted_index = np.argmax(predictions)

            # Validate prediction index
            if predicted_index < len(class_labels):
                predicted_class = class_labels[predicted_index]
                st.subheader(f"✅ Predicted Class: **{predicted_class}**")
            else:
                st.error("⚠️ Prediction index out of range. Check model and labels.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
