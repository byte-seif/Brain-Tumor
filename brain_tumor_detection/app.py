import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import nbformat
from nbconvert import HTMLExporter

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
    image = ImageOps.fit(image, (150, 150), Image.ANTIALIAS)
    img_array = np.asarray(image) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to display the notebook
def display_notebook(notebook_path):
    with open(notebook_path, "r") as f:
        notebook_content = f.read()
    notebook_node = nbformat.reads(notebook_content, as_version=4)
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'basic'
    (body, _) = html_exporter.from_notebook_node(notebook_node)
    st.components.v1.html(body, height=800, scrolling=True)

# Streamlit app layout and interactions
st.title("Brain Tumor Detection Project")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Model Testing", "View Notebook"])

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
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        confidence = np.max(predictions)
        predicted_class = class_labels[np.argmax(predictions)]

        # Display the prediction result and confidence level
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}")

elif page == "View Notebook":
    st.header("Project Notebook")
    display_notebook("brain_tumor_notebook.ipynb")
