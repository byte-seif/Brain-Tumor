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

        # Corrected to use `components.html()`
        components.html(body, height=1200, width=1200, scrolling=True)
        logging.info("Notebook content rendered successfully.")

    except Exception as e:
        st.error(f"An error occurred while displaying the notebook: {e}")
        logging.error(f"An error occurred while displaying the notebook: {e}")

# Streamlit app layout and interactions
st.title("Brain Tumor Detection and Classification")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Model Testing", "View Notebook"])

# Define Model Testing page
if page == "Model Testing":
    st.header("Brain Tumor Classification from MRI Scans")
    st.write("Upload an MRI image, and the model will classify it into one of the four categories.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        st.write("Classifying...")
        try:
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            confidence = np.max(predictions)
            predicted_class = class_labels[np.argmax(predictions)]

            st.write(f"**Prediction:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}")
        except ValueError as e:
            st.error(f"An error occurred during prediction: {e}")
            logging.error(f"An error occurred during prediction: {e}")

# Define Notebook View page
elif page == "View Notebook":
    st.header("Project Notebook")
    display_notebook("brain_tumor_detection/CNN.ipynb")






