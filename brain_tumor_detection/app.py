import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import nbformat
from nbconvert import HTMLExporter
import os
import streamlit.components.v1 as components
import logging

# Set Streamlit page configuration for wide layout and title
st.set_page_config(
    page_title="Brain Tumor Detection and Classification",
    layout="wide",  # Set the layout to wide
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename="app.log", filemode="w")

# Custom CSS to increase overall element size for a desktop display
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 20px;
        }
        .css-1v3fvcr {
            width: 350px;
        }
        .stButton button {
            font-size: 1.2em;
            padding: 15px 30px;
        }
        .stTextInput, .stTextArea {
            font-size: 1.2em;
            padding: 10px;
        }
        .css-1y0tads {
            font-size: 1.2em;
        }
        h1, h2, h3 {
            font-size: 2em !important;
        }
        .main .block-container {
            padding: 2rem 5rem;
        }
        .css-2trqyj {
            max-width: 90%;
        }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_detection/simplified_cnn_model.h5")
    return model

model = load_model()

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Preprocess the uploaded image
def preprocess_image(image: Image.Image) -> np.array:
    target_size = (150, 150)
    image = ImageOps.fit(image, target_size, Image.LANCZOS)

    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.asarray(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Function to display the notebook
def display_notebook(notebook_path):
    if not os.path.isfile(notebook_path):
        st.error(f"Notebook file '{notebook_path}' not found.")
        logging.error(f"Notebook file '{notebook_path}' not found.")
        return

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = f.read()
            logging.info("Notebook content read successfully.")

        notebook_node = nbformat.reads(notebook_content, as_version=4)
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'lab'
        (body, _) = html_exporter.from_notebook_node(notebook_node)
        logging.info("Notebook content converted to HTML successfully.")

        if not body:
            st.error("Notebook content could not be rendered. Please check the notebook file.")
            logging.error("Notebook content could not be rendered.")
            return

        components.html(body, height=1200, width=1200, scrolling=True)
        logging.info("Notebook content rendered successfully.")

    except Exception as e:
        st.error(f"An error occurred while displaying the notebook: {e}")
        logging.error(f"An error occurred while displaying the notebook: {e}")

# Streamlit app layout and interactions
st.title("Brain Tumor Detection and Classification")

# Project Background Information and Description
st.markdown("""
## **Background Information**

A brain tumor is an abnormal growth of cells in the brain or nearby tissues (e.g., meninges, cranial nerves, or pituitary gland) and can be primary (originating in the brain) or secondary (metastatic, spreading from other organs). Primary brain tumors are either benign or malignant, with common malignant types including gliomas (from glial cells), meningiomas (from meninges), pituitary tumors, and medulloblastomas (common in children).

Risk factors for brain tumors include genetic predisposition and exposure to ionizing radiation, though causes are often unknown. Symptoms vary based on tumor size, type, and location, typically manifesting as headaches, seizures, cognitive changes, or neurological deficits. MRI and CT scans are standard diagnostic tools, often followed by a biopsy to assess tumor type and grade.

### **Objective of the Project**

This project aims to develop a deep learning-based solution to automate brain tumor detection and classification using MRI images. By leveraging convolutional neural networks (CNNs), the model identifies and categorizes brain tumors into one of four classes, thereby assisting radiologists and clinicians in making faster, data-driven decisions for patient care. Such an AI-based solution can be a transformative tool, potentially reducing diagnostic time, minimizing inter-observer variability, and supporting more consistent clinical outcomes.

### **Tumor Classes in the Dataset**

1. **Glioma**: A type of malignant tumor originating from glial cells, which provide support and protection for neurons. Gliomas are often aggressive and require prompt intervention.

2. **Meningioma**: Generally benign, meningiomas originate in the meninges—the protective layers covering the brain and spinal cord. While less aggressive, they may cause significant health issues due to their location.

3. **No Tumor**: MRI scans in this category show no evidence of brain tumors, serving as a control to validate the model’s ability to distinguish healthy cases from pathological ones.

4. **Pituitary Tumor**: Located at the base of the brain, these tumors arise from the pituitary gland and can impact hormone production and neurological function. Most pituitary tumors are benign but may still require medical intervention.

### **Model Implementation**

For this project, a **Convolutional Neural Network (CNN)** was implemented to classify MRI images into the four defined categories. CNNs are particularly effective in image classification tasks due to their ability to learn hierarchical features through multiple layers. The model architecture includes a series of convolutional layers to capture spatial features, followed by fully connected layers to interpret these features for classification.

### **Project Goals**

1. **Accurate Tumor Classification**: Achieve high sensitivity and specificity in identifying and classifying brain tumors across four categories.
2. **Real-time Prediction**: Allow clinicians and users to upload MRI images for immediate classification and receive results with confidence scores.
3. **Support Clinical Decision-making**: Provide a consistent and efficient diagnostic tool to assist radiologists, ultimately contributing to faster patient management and improved outcomes.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Model Testing", "View Notebook"])

# Define Model Testing page
if page == "Model Testing":
    st.header("Brain Tumor Classification from MRI Scans")
    st.write("Upload an MRI image, and the model will classify it into one of the four categories.")

    # File uploader for user to upload MRI image
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        # Preprocess and classify the image
        st.write("Classifying...")
        try:
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            confidence = np.max(predictions)
            predicted_class = class_labels[np.argmax(predictions)]

            # Display the prediction result and confidence level
            st.write(f"**Prediction:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}")
        except ValueError as e:
            st.error(f"An error occurred during prediction: {e}")
            logging.error(f"An error occurred during prediction: {e}")

# Define Notebook View page
elif page == "View Notebook":
    st.header("Project Notebook")
    display_notebook("brain_tumor_detection/CNN.ipynb")







