import streamlit as st
import requests
import io
from PIL import Image

# Defining the API endpoint
API_URL = "http://api:80/predict/"

st.title("Glasses Detection App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Converting image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Sending image to API
    response = requests.post(API_URL, files={"file": img_byte_arr})
    
    if response.status_code == 200:
        result = response.json()
        st.write(f"Prediction: The person is {'wearing glasses' if result['is_wearing_glasses'] else 'not wearing glasses'}.")
    else:
        st.write("Error: Unable to classify image.")