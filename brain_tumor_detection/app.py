import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import nbformat
from nbconvert import HTMLExporter
import os

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
    # Resize the image with high-quality downsampling
    image = ImageOps.fit(image, (150, 150), Image.LANCZOS)
    img_array = np.asarray(image) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to display the notebook

def display_notebook(notebook_path):
    # Check if the notebook file exists
    if not os.path.isfile(notebook_path):
        st.error("Notebook file not found.")
        return

    # Read the notebook content
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_content = f.read()

    # Convert the notebook to HTML
    notebook_node = nbformat.reads(notebook_content, as_version=4)
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'lab'  # Use 'lab' template for better rendering
    (body, _) = html_exporter.from_notebook_node(notebook_node)

    # Display HTML in Streamlit
    st.components.v1.html(body, height=1000, scrolling=True)


# Streamlit app layout and interactions
st.title("Brain Tumor Detection and Classification")

st.markdown("""
## **Background**

Brain tumors represent abnormal cell growths within the brain or surrounding tissues, including the meninges, cranial nerves, and pituitary gland. These tumors can be either primary (originating within the brain) or secondary (metastatic, spreading from other parts of the body). Accurate diagnosis and timely intervention are essential, as brain tumors often pose significant health risks due to their rapid progression and impact on neurological function. Common brain tumor types include gliomas (arising from glial cells), meningiomas (from meninges), pituitary tumors, and others, each with varying prognoses and treatment strategies.

Magnetic Resonance Imaging (MRI) is the standard diagnostic tool for identifying brain tumors. However, manual analysis of MRI images can be time-intensive and prone to variability, highlighting the need for automated solutions to improve diagnostic accuracy and efficiency.

## **Objective of the Project**

This project aims to develop a deep learning-based solution to automate brain tumor detection and classification using MRI images. By leveraging convolutional neural networks (CNNs), the model identifies and categorizes brain tumors into one of four classes, thereby assisting radiologists and clinicians in making faster, data-driven decisions for patient care. Such an AI-based solution can be a transformative tool, potentially reducing diagnostic time, minimizing inter-observer variability, and supporting more consistent clinical outcomes.

## **Tumor Classes in the Dataset**

1. **Glioma**: A type of malignant tumor originating from glial cells, which provide support and protection for neurons. Gliomas are often aggressive and require prompt intervention.

2. **Meningioma**: Generally benign, meningiomas originate in the meninges—the protective layers covering the brain and spinal cord. While less aggressive, they may cause significant health issues due to their location.

3. **No Tumor**: MRI scans in this category show no evidence of brain tumors, serving as a control to validate the model’s ability to distinguish healthy cases from pathological ones.

4. **Pituitary Tumor**: Located at the base of the brain, these tumors arise from the pituitary gland and can impact hormone production and neurological function. Most pituitary tumors are benign but may still require medical intervention.

## **Model Implementation**

For this project, a **Convolutional Neural Network (CNN)** was implemented to classify MRI images into the four defined categories. CNNs are particularly effective in image classification tasks due to their ability to learn hierarchical features through multiple layers. The model architecture includes a series of convolutional layers to capture spatial features, followed by fully connected layers to interpret these features for classification.

## **Project Goals**

1. **Accurate Tumor Classification**: Achieve high sensitivity and specificity in identifying and classifying brain tumors across four categories.
2. **Real-time Prediction**: Allow clinicians and users to upload MRI images for immediate classification and receive results with confidence scores.
3. **Support Clinical Decision-making**: Provide a consistent and efficient diagnostic tool to assist radiologists, ultimately contributing to faster patient management and improved outcomes.

--- 

### **How to Use This App**

Upload an MRI image, and the model will classify it into one of the four classes with a corresponding confidence score. You can also explore the project notebook to understand the model development process and key insights from the data.
""")

# Sidebar navigation
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
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        confidence = np.max(predictions)
        predicted_class = class_labels[np.argmax(predictions)]

        # Display the prediction result and confidence level
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}")

# Define Notebook View page
elif page == "View Notebook":
    st.header("Project Notebook")
    display_notebook("brain_tumor_detection/brain_tumor_notebook.ipynb")


