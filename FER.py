import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('D:\AMIT\projects\FER Project\FER\my_model.keras')

# Function to load and preprocess the uploaded image
def prepare_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Resize the image
    
    # If the image is grayscale (2D), convert it to RGB (3D)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch size dimension
    return img_array

# Updated class names for emotions
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Streamlit interface
st.title('Emotion Classification with Deep Learning Model')

st.write(
    "This app allows you to upload an image and classify it into one of the following emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral."
)

# Upload image from the user
uploaded_file = st.file_uploader("Choose an image to classify", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Prepare the image
    img_array = prepare_image(uploaded_file)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_names[predicted_class[0]]

    st.write(f"Predicted Emotion: {predicted_label}")
    st.write(f"Prediction Confidence: {np.max(predictions) * 100:.2f}%")
