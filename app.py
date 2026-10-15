import os
import pickle
import numpy as np
import streamlit as st
from deepface import DeepFace
from PIL import Image

st.set_page_config(page_title="Indian Cricketer Face ID", layout="wide")
st.title("🏏 Indian Cricket Team Face Identification App")

@st.cache_data
def load_encodings():
    if os.path.exists("encodings.pkl"):
        with open("encodings.pkl", "rb") as f:
            return pickle.load(f)
    return {}

known_db = load_encodings()

uploaded_file = st.file_uploader("Upload an image of an Indian Cricketer", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temp uploaded image for processing
    with open("temp_upload.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    with col1:
        st.image("temp_upload.jpg", caption="Uploaded Image", use_container_width=True)

    with col2:
        try:
            with st.spinner("Analyzing facial features..."):
                results = DeepFace.find(
                    img_path="temp_upload.jpg", 
                    db_path="dataset", 
                    model_name="Facenet", 
                    enforce_detection=False
                )
                
            if len(results) > 0 and not results[0].empty:
                matched_path = results[0].iloc[0]['identity']
                predicted_name = os.path.basename(os.path.dirname(matched_path)).replace("_", " ").title()
                distance = results[0].iloc[0]['distance']
                confidence = max(0, min(100, (1 - distance) * 100))
                
                st.success(f"**Identified Player:** {predicted_name}")
                st.metric(label="Match Confidence", value=f"{confidence:.2f}%")
            else:
                st.error("No matching player found in database.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
