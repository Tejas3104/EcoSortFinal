import streamlit as st
import numpy as np
#import cv2
from keras.models import load_model
from PIL import Image
import io
import openai

# Set your OpenAI API key here
openai.api_key = "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"

# Load the model and labels
@st.cache_resource
def load_model_func():
    model = load_model('path/to/your/model.h5')
    return model

@st.cache_resource
def load_labels():
    with open('path/to/your/labels.txt') as f:
        labels = f.read().splitlines()
    return labels

# Preprocess the image
def preprocess_image(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((224, 224))  # Adjust size according to your model's input
    image_data = np.array(image) / 255.0  # Normalize the image
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
    return image_data

# Classify the image
def classify_image(model, labels, image_data):
    predictions = model.predict(image_data)
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    return labels[predicted_label_index]

# Function to get suggestions based on the predicted label
def get_suggestions(predicted_label):
    suggestions_dict = {
        "cardboard": [
            "Flatten the box before recycling.",
            "Remove any tape or plastic windows."
        ],
        "plastic": [
            "Rinse out containers before recycling.",
            "Check for the recycling symbol on the bottom."
        ],
        "glass": [
            "Rinse glass bottles and jars.",
            "Remove any metal lids or caps."
        ],
        "metal": [
            "Clean food containers before recycling.",
            "Check for local recycling guidelines."
        ],
        "paper": [
            "Keep paper dry and clean for recycling.",
            "Shred confidential documents before recycling."
        ],
        "trash": [
            "Dispose of non-recyclable items in the trash.",
            "Consider composting organic waste."
        ],
        "compost": [
            "Add kitchen scraps and yard waste.",
            "Avoid adding meat or dairy."
        ],
    }
    return suggestions_dict.get(predicted_label, [])

# Function to get additional insights from OpenAI
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# Streamlit app layout
def show_classification_page():
    st.markdown('<div class="header-title">EcoSort</div>', unsafe_allow_html=True)
    st.write("Select an option to classify waste:")

    # Add radio button for choosing the input method
    option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

    # Load the model and labels when the app starts
    model, labels = None, None

    try:
        model = load_model_func()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

    try:
        labels = load_labels()
        st.success("Labels loaded successfully!")
    except Exception as e:
        st.error(f"Error loading labels: {e}")

    # Handle image upload
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            st.write("")

            # Preprocess the image and make predictions using the model
            image_data = preprocess_image(uploaded_file)
            if model and labels:
                predicted_label = classify_image(model, labels, image_data)
                st.write(f"Predicted label: **{predicted_label}**", unsafe_allow_html=True)
                print(f"Predicted label: {predicted_label}")  # Debug output

                # Get recycling suggestions
                suggestions = get_suggestions(predicted_label)
                st.subheader("Recycling Suggestions:")
                if suggestions:
                    for suggestion in suggestions:
                        st.markdown(f'<div class="suggestion">{suggestion}</div>', unsafe_allow_html=True)
                else:
                    st.write("No suggestions available.")

                # Optional: Use OpenAI API to get more insights
                openai_response = get_openai_response(f"Provide insights on recycling for {predicted_label}.")
                st.subheader("Additional Insights:")
                if openai_response:
                    st.write(openai_response)
                else:
                    st.write("No insights returned from OpenAI.")

            else:
                st.error("Model or labels not available. Please check if they were loaded correctly.")

    # Handle webcam capture
    if option == "Use Webcam":
        camera_input = st.camera_input("Take a picture")
        
        if camera_input is not None:
            # Display the captured image
            st.image(camera_input, caption='Captured Image', use_column_width=True)
            st.write("")

            # Preprocess the image and make predictions using the model
            image_data = preprocess_image(camera_input)
            if model and labels:
                predicted_label = classify_image(model, labels, image_data)
                st.write(f"Predicted label: **{predicted_label}**", unsafe_allow_html=True)
                print(f"Predicted label: {predicted_label}")  # Debug output

                # Get recycling suggestions
                suggestions = get_suggestions(predicted_label)
                st.subheader("Recycling Suggestions:")
                if suggestions:
                    for suggestion in suggestions:
                        st.markdown(f'<div class="suggestion">{suggestion}</div>', unsafe_allow_html=True)
                else:
                    st.write("No suggestions available.")

                # Optional: Use OpenAI API to get more insights
                openai_response = get_openai_response(f"Provide insights on recycling for {predicted_label}.")
                st.subheader("Additional Insights:")
                if openai_response:
                    st.write(openai_response)
                else:
                    st.write("No insights returned from OpenAI.")

            else:
                st.error("Model or labels not available. Please check if they were loaded correctly.")

# Run the Streamlit app
if __name__ == "__main__":
    show_classification_page()
