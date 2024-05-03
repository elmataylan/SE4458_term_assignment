import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Note: these codes work good in google Colab

# Function to load and preprocess image
def load_and_preprocess_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))  # Resize to match input shape of MobileNetV2
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to classify image
def classify_image(url):
    img = load_and_preprocess_image(url)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    preds = model.predict(img)
    decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print("{}. {}: {:.2f}%".format(i + 1, label, score * 100))

# Example image URL
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Gorille_des_plaines_de_l%27ouest_%C3%A0_l%27Espace_Zoologique.jpg/800px-Gorille_des_plaines_de_l%27ouest_%C3%A0_l%27Espace_Zoologique.jpg"

# Classify image
classify_image(image_url)