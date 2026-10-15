import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Indian Cricketer Face ID", layout="wide")
st.title("🏏 Indian Cricket Team Face Identification App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Load OpenCV's lightweight Haar Cascade face detector
@st.cache_resource
def load_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

# Extract lightweight color histograms as feature vectors
def extract_face_features(image_np, face_box):
    x, y, w, h = face_box
    face_roi = image_np[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (100, 100))
    # Compute normalized HSV color/texture histogram
    hsv = cv2.cvtColor(face_resized, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

@st.cache_data
def build_lightweight_database():
    detector = load_detector()
    db_features = []
    db_labels = []
    
    if not os.path.exists(DATASET_DIR):
        return db_features, db_labels

    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                pil_img = Image.open(img_path).convert("RGB")
                img_np = np.array(pil_img)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                
                if len(faces) > 0:
                    feat = extract_face_features(img_np, faces[0])
                    db_features.append(feat)
                    db_labels.append(person_name.replace("_", " ").title())
            except Exception:
                continue

    return np.array(db_features), db_labels

if not os.path.exists(DATASET_DIR):
    st.error(f"Dataset folder not found at `{DATASET_DIR}`. Please ensure the dataset folder is committed to GitHub.")
else:
    with st.spinner("Initializing lightweight face database..."):
        db_features, db_labels = build_lightweight_database()

    st.sidebar.success(f"Indexed {len(db_features)} face samples.")

    uploaded_file = st.file_uploader("Upload an image of an Indian Cricketer", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(pil_image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)

        with col2:
            detector = load_detector()
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            if len(faces) == 0:
                st.warning("No faces detected in the uploaded image.")
            else:
                input_feat = extract_face_features(img_np, faces[0])
                
                # Find closest match using Chi-Square distance
                distances = [cv2.compareHist(input_feat, db_feat, cv2.HISTCMP_CHISQR) for db_feat in db_features]
                best_idx = np.argmin(distances)
                min_dist = distances[best_idx]

                predicted_name = db_labels[best_idx]
                confidence = max(0, min(100, (1 - (min_dist / 10.0)) * 100))

                st.success(f"**Identified Player:** {predicted_name}")
                st.metric(label="Match Confidence", value=f"{confidence:.1f}%")
