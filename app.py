import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Google Drive Model URL
drive_url = "https://drive.google.com/file/d/1os3m_b2PYcvCz33Ku_vzRjSkhBh8Y7e5"
model_path = "mobilenetv2_model.h5"

# Function to download the model from Google Drive
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.write("Downloading model... ⏳")
        gdown.download(drive_url, model_path, quiet=False)

    try:
        model = tf.keras.models.load_model(model_path)
        st.write("✅ Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Define class labels
class_labels = [
    "a_Good", "b_Moderate", "c_Unhealthy_for_Sensitive_Groups",
    "d_Unhealthy", "e_Very_Unhealthy", "f_Severe"
]

# Air quality descriptions
aq_descriptions = {
    "a_Good": {"Range": "0-50", "Label": "Good", "Description": "Enjoy your usual outdoor activities"},
    "b_Moderate": {"Range": "51-100", "Label": "Moderate", "Description": "Extremely sensitive children and adults should refrain from strenuous outdoor activities."},
    "c_Unhealthy_for_Sensitive_Groups": {"Range": "101-150", "Label": "Unhealthy for Sensitive Groups", "Description": "Sensitive children and adults should limit prolonged outdoor activity."},
    "d_Unhealthy": {"Range": "151-200", "Label": "Unhealthy", "Description": "Sensitive groups should avoid outdoor exposure and others should limit prolonged outdoor activity."},
    "e_Very_Unhealthy": {"Range": "201-300", "Label": "Very Unhealthy", "Description": "Sensitive groups should stay indoors and others should avoid outdoor activity."},
    "f_Severe": {"Range": "301-500", "Label": "Hazardous", "Description": "Everyone should stay indoors and avoid physical activity."}
}

# Streamlit UI
st.title("🌍 Air Quality Classification")
st.write("Upload an image to classify air quality.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

def preprocess_image(image):
    """Preprocess the image for model prediction."""
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image  # Shape: (1, 224, 224, 3)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Classify"):
        if model is None:
            st.error("⚠️ Model is not loaded. Please check for errors.")
        else:
            try:
                # Load and preprocess the image
                image = load_img(uploaded_file, target_size=(224, 224))
                image_preprocessed = preprocess_image(image)

                # Make prediction
                predictions = model.predict(image_preprocessed)
                
                # Debugging: Print the predictions array
                st.write("🔢 Predictions array:", predictions)

                if predictions.size == 0:
                    st.error("⚠️ No predictions were made. Please check the model and input image.")
                else:
                    predicted_class_index = np.argmax(predictions)

                    if predicted_class_index >= len(class_labels):
                        st.error(f"⚠️ Predicted class index {predicted_class_index} is out of range.")
                    else:
                        predicted_class = class_labels[predicted_class_index]
                        result = aq_descriptions[predicted_class]

                        # Display results
                        st.subheader(f"📌 Prediction: {result['Label']} ({result['Range']})")
                        st.write(result['Description'])

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
