import os
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Indian Cricketer Face ID", layout="wide")
st.title("🏏 Indian Cricket Team Face Identification App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Convert image into lightweight feature vector using PIL & Numpy (No OpenCV required)
def extract_pil_features(pil_img):
    img_resized = pil_img.resize((64, 64)).convert("L")  # Convert to 64x64 Grayscale
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return img_array.flatten()

@st.cache_resource
def train_lightweight_model():
    X_train = []
    y_train = []
    
    if not os.path.exists(DATASET_DIR):
        return None, []

    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                pil_img = Image.open(img_path)
                feat = extract_pil_features(pil_img)
                X_train.append(feat)
                y_train.append(person_name.replace("_", " ").title())
            except Exception:
                continue

    if len(X_train) == 0:
        return None, []

    # Train a fast, low-memory K-Nearest Neighbors Classifier
    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn.fit(X_train, y_train)
    return knn, list(set(y_train))

if not os.path.exists(DATASET_DIR):
    st.error(f"Dataset folder not found at `{DATASET_DIR}`. Please verify that your `dataset` folder is committed to GitHub.")
else:
    with st.spinner("Training lightweight model..."):
        model, classes = train_lightweight_model()

    if model is None:
        st.error("No valid image data found in dataset directory.")
    else:
        st.sidebar.success(f"Model trained successfully on {len(classes)} players!")

        uploaded_file = st.file_uploader("Upload an image of an Indian Cricketer", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_image, caption="Uploaded Image", use_container_width=True)

            with col2:
                try:
                    test_feat = extract_pil_features(pil_image).reshape(1, -1)
                    prediction = model.predict(test_feat)[0]
                    probs = model.predict_proba(test_feat)[0]
                    confidence = np.max(probs) * 100

                    st.success(f"**Identified Player:** {prediction}")
                    st.metric(label="Match Confidence", value=f"{confidence:.1f}%")
                except Exception as e:
                    st.error(f"Error predicting image: {e}")
