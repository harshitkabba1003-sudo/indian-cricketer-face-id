import os
import streamlit as st
from deepface import DeepFace

st.set_page_config(page_title="Indian Cricketer Face ID", layout="wide")
st.title("🏏 Indian Cricket Team Face Identification App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

if not os.path.exists(DATASET_DIR):
    st.error(f"Dataset folder not found at `{DATASET_DIR}`. Please make sure the `dataset` folder exists in your GitHub repo.")
else:
    uploaded_file = st.file_uploader("Upload an image of an Indian Cricketer", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_img_path = os.path.join(BASE_DIR, "temp_upload.jpg")
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)
        with col1:
            st.image(temp_img_path, caption="Uploaded Image", use_container_width=True)

        with col2:
            try:
                with st.spinner("Analyzing facial features..."):
                    # Using Fast Facenet + OpenCV face detector
                    results = DeepFace.find(
                        img_path=temp_img_path, 
                        db_path=DATASET_DIR, 
                        model_name="Facenet",
                        detector_backend="opencv",
                        enforce_detection=False
                    )
                    
                if len(results) > 0 and not results[0].empty:
                    matched_path = results[0].iloc[0]['identity']
                    # Extracts folder name (e.g., jasprit_bumrah -> Jasprit Bumrah)
                    predicted_name = os.path.basename(os.path.dirname(matched_path)).replace("_", " ").title()
                    distance = results[0].iloc[0]['distance']
                    confidence = max(0, min(100, (1 - distance) * 100))
                    
                    st.success(f"**Identified Player:** {predicted_name}")
                    st.metric(label="Match Confidence", value=f"{confidence:.2f}%")
                else:
                    st.error("No matching player found in database.")
            except Exception as e:
                st.error(f"Error processing image: {e}")
