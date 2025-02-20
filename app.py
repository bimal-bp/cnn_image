import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Check if the model file exists
model_path = "resnet_vit_model.h5"
model = None  # Initialize model variable

if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Model file '{model_path}' not found!")

# Define class labels
class_labels = [
    "a_Good", "b_Moderate", "c_Unhealthy_for_Sensitive_Groups",
    "d_Unhealthy", "e_Very_Unhealthy", "f_Severe"
]

# UI with Streamlit
st.title("Air Quality Classification")
st.write("Upload an image to classify air quality.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

def preprocess_image(image):
    """Preprocess image for both ResNet and ViT (same format)."""
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
                # Load and resize the image
                image = load_img(uploaded_file, target_size=(224, 224))

                # Preprocess image
                image_preprocessed = preprocess_image(image)

                # Make prediction
                predictions = model.predict(image_preprocessed)
                predicted_class = class_labels[np.argmax(predictions)]

                # Air quality descriptions
                aq_descriptions = {
                    "a_Good": {"Range": "0-50", "Label": "Good", "Description": "Enjoy your usual outdoor activities"},
                    "b_Moderate": {"Range": "51-100", "Label": "Moderate", "Description": "Extremely sensitive children and adults should refrain from strenuous outdoor activities."},
                    "c_Unhealthy_for_Sensitive_Groups": {"Range": "101-150", "Label": "Unhealthy for Sensitive Groups", "Description": "Sensitive children and adults should limit prolonged outdoor activity."},
                    "d_Unhealthy": {"Range": "151-200", "Label": "Unhealthy", "Description": "Sensitive groups should avoid outdoor exposure and others should limit prolonged outdoor activity."},
                    "e_Very_Unhealthy": {"Range": "201-300", "Label": "Very Unhealthy", "Description": "Sensitive groups should stay indoors and others should avoid outdoor activity."},
                    "f_Severe": {"Range": "301-500", "Label": "Hazardous", "Description": "Everyone should stay indoors and avoid physical activity."}
                }

                result = aq_descriptions[predicted_class]

                # Display results
                st.subheader(f"Prediction: {result['Label']} ({result['Range']})")
                st.write(result['Description'])
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
