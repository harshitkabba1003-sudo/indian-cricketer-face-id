import os
import pickle
from deepface import DeepFace

DATASET_DIR = "dataset"
encodings_db = {}

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    
    encodings_db[person_name] = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        try:
            # Extract fast 128D/512D facial embedding representation
            embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            encodings_db[person_name].append(embedding)
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

with open("encodings.pkl", "wb") as f:
    pickle.dump(encodings_db, f)

print("Saved encodings.pkl successfully!")
