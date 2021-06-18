import tensorflow as tf
import cv2
import numpy as np

CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value

#preparing image
def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based

    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("my_model.h5.model")

prediction = model.predict([prepare('predict.jpg')])
print((prediction[0][0]) * 100)
print(CATEGORIES[int(prediction[0][0])])
