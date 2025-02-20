import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown  # For downloading files from Google Drive
import os

# Download the model from Google Drive
drive_url = "https://drive.google.com/uc?id=1MGBH4qECimwgJGXLuEv2Y_ZEUV9b0Yql"
model_path = "mobilenetv2_model.h5"

# Download the model if it doesn't already exist
if not os.path.exists(model_path):
    gdown.download(drive_url, model_path, quiet=False)

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ['a_Good', 'b_Moderate', 'c_Unhealthy_for_Sensitive_Groups', 'd_Unhealthy', 'e_Very_Unhealthy', 'f_Severe']

# UI with Streamlit
st.title("Air Quality Classification")
st.write("Upload an image to classify air quality.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

def preprocess_image(image):
    """Preprocess image for the model."""
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image  # Shape: (1, 224, 224, 3)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify"):
        # Load and resize the image
        image = load_img(uploaded_file, target_size=(224, 224))

        # Preprocess the image
        image_processed = preprocess_image(image)

        # Make prediction
        predictions = model.predict(image_processed)
        predicted_index = np.argmax(predictions)

        # Check if the predicted index is within the range of class_labels
        if predicted_index < len(class_labels):
            predicted_class = class_labels[predicted_index]
        else:
            st.error("Predicted class index is out of range. Please check the model and class labels.")
            st.stop()

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
