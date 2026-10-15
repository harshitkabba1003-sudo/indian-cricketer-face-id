import os
import numpy as np
import face_recognition
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Indian Cricketer Face ID", layout="wide")
st.title("🏏 Indian Cricket Team Face Identification App")

DATASET_DIR = "dataset"

@st.cache_resource
def load_and_encode_dataset():
    known_encodings = []
    known_names = []
    
    if not os.path.exists(DATASET_DIR):
        return known_encodings, known_names

    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                # Load image via PIL instead of OpenCV
                pil_img = Image.open(img_path).convert("RGB")
                img_array = np.array(pil_img)
                encodings = face_recognition.face_encodings(img_array)
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name.replace("_", " ").title())
            except Exception:
                continue
                
    return known_encodings, known_names

st.sidebar.header("Dataset Status")
with st.spinner("Loading dataset facial encodings..."):
    known_encodings, known_names = load_and_encode_dataset()

st.sidebar.success(f"Loaded {len(known_encodings)} face samples across {len(set(known_names))} players.")

uploaded_file = st.file_uploader("Upload an image of an Indian Cricketer", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open uploaded file with PIL
    pil_image = Image.open(uploaded_file).convert("RGB")
    rgb_image = np.array(pil_image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    with col2:
        if len(face_encodings) == 0:
            st.warning("No facial features detected in the uploaded image.")
        else:
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown Person"

                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        confidence = (1 - face_distances[best_match_index]) * 100
                        st.success(f"**Identified Player:** {name}")
                        st.metric(label="Match Confidence", value=f"{confidence:.2f}%")
                    else:
                        st.error("Face detected, but does not match any player in database.")
