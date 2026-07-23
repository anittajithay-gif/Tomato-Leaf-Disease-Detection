 Step 1: Import Libraries
# ==========================

import gradio as gr
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2

print("✅ Libraries Imported Successfully")
print("TensorFlow Version:", tf.__version__)
# ==========================
# Step 2: Load Model
# ==========================

# Load trained tomato disease classifier
classifier = load_model(
    "tomato_disease_mobilenetv2.keras"
)

print("✅ Tomato Disease Classifier Loaded")


# ==========================
# Load MobileNetV2 Feature Extractor
# ==========================

feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

# Freeze feature extractor
feature_extractor.trainable = False

print("✅ MobileNetV2 Feature Extractor Loaded")
# ==========================
# Step 3: Tomato Disease Class Names
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


print("✅ Class Names Added")
print("Number of Classes:", len(class_names))
# ==========================
# Step 4: Disease Information Dictionary
# ==========================

disease_info = {

    "Tomato___Bacterial_spot": {
        "name": "Bacterial Spot",
        "description": "A bacterial disease that creates dark spots and lesions on tomato leaves.",
        "symptoms": "Small dark spots, yellowing around spots, and leaf damage.",
        "prevention": "Use disease-free seeds and remove infected plant parts.",
        "management": "Apply recommended copper-based sprays and maintain field hygiene."
    },


    "Tomato___Early_blight": {
        "name": "Early Blight",
        "description": "A fungal disease causing brown circular spots with ring patterns on leaves.",
        "symptoms": "Brown spots, yellow leaves, and early leaf dropping.",
        "prevention": "Avoid overhead watering and maintain proper plant spacing.",
        "management": "Remove infected leaves and apply suitable fungicides."
    },


    "Tomato___Late_blight": {
        "name": "Late Blight",
        "description": "A serious fungal disease affecting leaves, stems, and fruits.",
        "symptoms": "Dark irregular patches and rapid plant damage.",
        "prevention": "Reduce moisture and improve air circulation.",
        "management": "Use recommended fungicides and remove infected plants."
    },


    "Tomato___Leaf_Mold": {
        "name": "Leaf Mold",
        "description": "A fungal disease commonly appearing under high humidity conditions.",
        "symptoms": "Yellow patches on upper leaves and mold growth underneath.",
        "prevention": "Control humidity and provide proper ventilation.",
        "management": "Use suitable fungicides and remove infected leaves."
    },


    "Tomato___Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "description": "A fungal infection producing small circular spots on leaves.",
        "symptoms": "Dark bordered spots mainly on older leaves.",
        "prevention": "Remove infected leaves and avoid water contact on foliage.",
        "management": "Apply fungal control treatments."
    },


    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Spider Mites",
        "description": "Small pests that damage leaves by sucking plant nutrients.",
        "symptoms": "Yellow spots, leaf drying, and fine webbing.",
        "prevention": "Monitor plants regularly and maintain proper irrigation.",
        "management": "Use appropriate mite control methods."
    },


    "Tomato___Target_Spot": {
        "name": "Target Spot",
        "description": "Fungal disease producing circular target-like spots.",
        "symptoms": "Brown circular lesions on leaves and plant weakness.",
        "prevention": "Use resistant varieties and maintain plant health.",
        "management": "Apply recommended fungicides."
    },

     "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Yellow Leaf Curl Virus",
        "description": "Viral disease causing curling and yellowing of leaves.",
        "symptoms": "Leaf curling, yellow leaves, and reduced growth.",
        "prevention": "Control whiteflies and remove infected plants.",
        "management": "Use virus-resistant varieties."
    },


    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description": "Virus causing mosaic patterns on tomato leaves.",
        "symptoms": "Light and dark green patches on leaves.",
        "prevention": "Use clean tools and healthy seeds.",
        "management": "Remove infected plants and prevent virus spread."
    },


    "Tomato___healthy": {
        "name": "Healthy Tomato Leaf",
        "description": "The leaf is healthy without disease symptoms.",
        "symptoms": "Normal green leaf appearance.",
        "prevention": "Maintain proper watering and nutrition.",
        "management": "Continue regular crop monitoring."
    }

}


print("✅ Professional Disease Information Added")

# ==========================
# Step 5: Prediction Function
# ==========================

def predict_tomato(image):

    # Check image
    if image is None:
        return (
            "No Image",
            "0%",
            "Please upload a tomato leaf image.",
            "-",
            "-",
            "-"
        )


    # Convert image format
    image = cv2.resize(image, (224, 224))


    # Convert to float
    image = image.astype("float32")


    # Normalize image for MobileNetV2
    image = image / 255.0


    # Add batch dimension
    image = np.expand_dims(image, axis=0)


    # Feature extraction
    features = feature_extractor.predict(
        image,
        verbose=0
    )


    # Disease prediction
    prediction = classifier.predict(
        features,
        verbose=0
    )


    # Get predicted class
    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]


    # Confidence score
    confidence = float(
        np.max(prediction) * 100
    )


    # Get disease details
    info = disease_info[predicted_class]


    return (
        info["name"],
        f"{confidence:.2f}%",
        info["description"],
        info["symptoms"],
        info["prevention"],
        info["management"]
    )
# Step 6: Professional Gradio Blocks UI
# ==========================

import gradio as gr


with gr.Blocks(
    title="AI Tomato Leaf Disease Detection"
) as app:


    # Header Section
    gr.HTML("""
    <div style="text-align:center">

        <h1> AI-Based Tomato Leaf Disease Detection System</h1>

        <p style="font-size:18px;">
        Deep Learning powered plant disease identification
        using MobileNetV2 Transfer Learning
        </p>

    </div>
    """)


    gr.Markdown("---")


    # Main Prediction Section
    with gr.Row():

        # Image Upload Section
        with gr.Column(scale=1):

            input_image = gr.Image(
                type="numpy",
                 label="📷 Upload Tomato Leaf Image"
            )


            predict_btn = gr.Button(
                "🔍 Detect Disease",
                variant="primary"
            )


        # Output Section
        with gr.Column(scale=1):

            disease_output = gr.Textbox(
                label=" Predicted Disease"
            )


            confidence_output = gr.Textbox(
                label=" Confidence Score"
            )


            description_output = gr.Textbox(
                label=" Disease Description",
                lines=3
            )


            symptoms_output = gr.Textbox(
                label=" Symptoms",
                lines=3
            )


            prevention_output = gr.Textbox(
                label=" Prevention",
                 lines=3
            )


            management_output = gr.Textbox(
                label=" Management",
                lines=3
            )



    gr.Markdown("---")


    


    gr.Markdown("---")


    # Project Details Section

    gr.Markdown("""
    ##  Project Details


    **Project Title:**  
     AI-Based Tomato Leaf Disease Detection System


    **Technology Used:**  
    Deep Learning | MobileNetV2 Transfer Learning | TensorFlow | Gradio


    **Model:**  
    MobileNetV2 CNN Architecture

 **Developed By:**  
    Y Anitt Ajitha


    **Project Type:**  
    AI & ML Project - 2026


    """)



    # Button Connection

    predict_btn.click(
        fn=predict_tomato,
        inputs=input_image,
        outputs=[
            disease_output,
            confidence_output,
            description_output,
            symptoms_output,
            prevention_output,
            management_output
        ]
    )



# ==========================
# Launch Application
# ==========================

app.launch(server_name="0.0.0.0", server_port=10000)
