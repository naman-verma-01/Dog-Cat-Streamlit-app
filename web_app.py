
app.py

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image, ImageOps

CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value

model = tf.keras.models.load_model("/my_model.h5")


def prepare(image):
    IMG_SIZE = 50  # 50 in txt-based

    # img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def import_and_predict(image_data, model):
    size = (50, 50)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img.reshape(-1, 50, 50, 1)
    prediction = model.predict(img)
    return prediction


value = st.sidebar.radio('Radio', ["Main App", "About"])
if value == "Main App":
    st.title("AI Convolutional Neural Network")

    st.text("Dog vs Cat")

    image = st.file_uploader('Upload an image of a dog or a cat for the model to predict...', type=["jpg"])
    if image is None:
        st.text("Please upload a jpg")

    else:
        image = Image.open(image)

    if st.button("Calibrate..") and image is not None:
        prediction = import_and_predict(image, model)
        st.text(CATEGORIES[int(prediction[0][0])])
        st.success("Thank you for trying our model..")
    else:
        st.error("Please upload an image first")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

    if st.button("Exit"):
        st.stop()
elif value == "About":
    st.title("About")
    st.text("Author : Naman Verma and Team.")
    st.text(" ")
    st.text("About us..")
    st.text("""We are a team of enthusiastic programmers trying make world a 
          better place.""")
    st.text("Hope you enjoy our app :-)")
else:
    pass
