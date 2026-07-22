import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2


# ==========================
# Load Trained Classifier
# ==========================

classifier = load_model("tomato_disease_mobilenetv2.keras")


# ==========================
# Load Feature Extractor
# ==========================

feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

feature_extractor.trainable = False


# ==========================
# Class Names
# ==========================

class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ==========================
# Disease Information
# ==========================

disease_info = {

    "Tomato___Bacterial_spot": {
        "description": "Bacterial disease causing dark spots on tomato leaves.",
        "prevention": "Use disease-free seeds and apply copper-based fungicides."
    },

    "Tomato___Early_blight": {
        "description": "Fungal disease causing concentric brown spots on leaves.",
        "prevention": "Remove infected leaves and spray fungicide."
    },

    "Tomato___Late_blight": {
        "description": "Serious disease causing dark lesions on leaves and fruits.",
        "prevention": "Avoid excess moisture and apply recommended fungicides."
    },

    "Tomato___Leaf_Mold": {
        "description": "Leaf mold causes yellow patches on leaves.",
        "prevention": "Reduce humidity and improve air circulation."
    },

    "Tomato___Septoria_leaf_spot": {
        "description": "Small circular brown spots appear on older leaves.",
        "prevention": "Remove infected leaves and use fungicides."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Tiny mites suck plant sap and damage leaves.",
        "prevention": "Spray miticide and maintain proper irrigation."
    },

    "Tomato___Target_Spot": {
        "description": "Brown circular spots develop on tomato leaves.",
        "prevention": "Use resistant varieties and fungicides."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Virus causing yellow curled leaves and stunted growth.",
        "prevention": "Control whiteflies and remove infected plants."
    },

    "Tomato___Tomato_mosaic_virus": {
        "description": "Virus causing mosaic patterns on tomato leaves.",
        "prevention": "Use clean tools and resistant varieties."
    },

    "Tomato___healthy": {
        "description": "The tomato leaf is healthy and free from disease.",
        "prevention": "Continue proper watering and regular monitoring."
    }
}


# ==========================
# Prediction Function
# ==========================

def predict(image):

    image = cv2.resize(image, (224, 224))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)


    features = feature_extractor.predict(image, verbose=0)

    prediction = classifier.predict(features, verbose=0)


    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]


    confidence = float(np.max(prediction) * 100)


    description = disease_info[predicted_class]["description"]

    prevention = disease_info[predicted_class]["prevention"]


    disease_name = predicted_class.replace(
        "Tomato___", ""
    ).replace("_", " ")


    return f"""
🍅 TOMATO DISEASE DETECTION RESULT

✅ Disease:
{disease_name}

📊 Confidence:
{confidence:.2f}%

📝 Description:
{description}

💡 Prevention:
{prevention}
"""


# ==========================
# Gradio Interface
# ==========================

interface = gr.Interface(

    fn=predict,

    inputs=gr.Image(
        type="numpy",
        label="Upload Tomato Leaf Image"
    ),

    outputs=gr.Textbox(
        label="Prediction Result"
    ),

    title="🍅 AI-Based Tomato Leaf Disease Detection System",

    description="""
Upload a tomato leaf image.

This AI model will:

✅ Detect the disease

✅ Display confidence score

✅ Show disease description

✅ Suggest prevention methods
""",

    article="""
---
### 👩‍💻 Developed By

**Y Anitt Ajitha**

**Internship Project - 2026**

**Deep Learning Model: MobileNetV2**
"""
)


interface.launch()