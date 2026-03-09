import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


st.set_page_config(page_title="Sports Classifier", layout="centered")

@st.cache_resource
def load_model():

    return tf.keras.models.load_model('models/sports_classifier.h5')

model = load_model()

train_dir = 'data/train'

try:
    class_name = sorted(os.listdir(train_dir))
except FileNotFoundError:
    st.error("File not found")
    st.stop()

st.title("Sports classifier from image. Using CNN")

uploaded_file = st.file_uploader("choose an image",type= ['JPG','JPEG','PNG'])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image,caption='uploaded image',use_column_width=True)

    with st.spinner("Classifying..."):
        image = image.convert('RGB')
        image = image.resize((224,224))

        img_array = np.array(image)
        img_array = np.expand_dims(img_array,axis=0)

        pred = model.predict(img_array)
        pred_idx = np.argmax(pred[0])
        confidence = np.max(pred[0]) * 100

        predicted_sport = class_name[pred_idx].replace('_',' ')

        st.success(f"Prediction: {predicted_sport.title()}")
        st.info(f"Confidence: {confidence:.2f}%")