import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('cifar10_model.h5')
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
def preprocess_image(image):
    image = image.resize((32, 32))  
    image = np.array(image) 
    image = image / 255.0   
    image = np.expand_dims(image, axis=0)  
    return image

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and let the model predict its class.")


uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:

    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    st.write(f"Predicted Class: **{class_names[predicted_class]}**")