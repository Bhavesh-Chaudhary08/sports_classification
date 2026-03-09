import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.set_page_config(page_title="Sports Classifier", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('sports_classifier.h5')

model = load_model()

# Read classes from the JSON file instead of the missing data directory
try:
    with open('sports_label.json', 'r') as f:
        class_dict = json.load(f)
        # Convert the JSON dictionary into a list ordered by the numeric keys
        class_name = [class_dict[str(i)] for i in range(len(class_dict))]
except FileNotFoundError:
    st.error("sports_label.json not found! Please upload it to your GitHub repository.")
    st.stop()

st.title("Sports classifier from image. Using CNN")

uploaded_file = st.file_uploader("choose an image", type=['JPG', 'JPEG', 'PNG'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='uploaded image', use_column_width=True)

    with st.spinner("Classifying..."):
        image = image.convert('RGB')
        image = image.resize((224, 224))

        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        pred = model.predict(img_array)
        pred_idx = np.argmax(pred[0])
        confidence = np.max(pred[0]) * 100

        # Map prediction index to the class name
        predicted_sport = class_name[pred_idx].replace('_', ' ')

        st.success(f"Prediction: {predicted_sport.title()}")
        st.info(f"Confidence: {confidence:.2f}%")
