# indian-cricketer-face-id
# 🏏 Indian Cricket Team Member Face Identification

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://indian-cricketer-face-id-2edr8gd92mfh6j6eztbyaq.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end Computer Vision and Machine Learning web application designed to automatically detect and identify Indian Cricket Team members from uploaded images using facial feature extraction.
🏏 Indian Cricket Team Member Face Identification
An end-to-end Computer Vision and Machine Learning web application designed to automatically detect and identify Indian Cricket Team members from uploaded images using facial feature extraction.
Steamlit link : https://indian-cricketer-face-id-2edr8gd92mfh6j6eztbyaq.streamlit.app/

📸 Dataset & Supported Players
The system is trained and configured to recognize facial encodings for 14 Indian Cricket Players[cite: 1]:

🧢 Dinesh Karthik[cite: 1]

⚡ Hardik Pandya[cite: 1]

🎯 Jasprit Bumrah[cite: 1]

🏏 K. L. Rahul[cite: 1]

🧢 Kedar Jadhav[cite: 1]

🌀 Kuldeep Yadav[cite: 1]

⚡ Mohammed Shami[cite: 1]

👑 MS Dhoni[cite: 1]

⚔️ Ravindra Jadeja[cite: 1]

💥 Rohit Sharma[cite: 1]

🏏 Shikhar Dhawan[cite: 1]

🧢 Vijay Shankar[cite: 1]

👑 Virat Kohli[cite: 1]

🌀 Yuzvendra Chahal[cite: 1]

🧪 How to Test the Live Application
You can test the face identification web app without installing any software or running local code:

Open the Web App: Click the Streamlit badge above (or navigate to [https://your-app-url.streamlit.app](https://your-app-url.streamlit.app)).

Download a Test Image: Search for any clear face photograph of one of the 14 supported players listed above on Google Images.

Upload the Image:

Drag and drop the .jpg, .jpeg, or .png image into the upload box on the webpage.

Or click Browse files and select your downloaded image.

View Results:

The model will process the image, locate the face, extract facial embeddings, and display the predicted player's name alongside a Match Confidence Percentage.

📁 Repository Structure
Plaintext
├── dataset/
│   ├── dinesh_karthik/
│   ├── hardik_pandya/
│   ├── jasprit_bumrah/
│   ├── k._l._rahul/
│   ├── kedar_jadhav/
│   ├── kuldeep_yadav/
│   ├── mohammed_shami/
│   ├── ms_dhoni/
│   ├── ravindra_jadeja/
│   ├── rohit_sharma/
│   ├── shikhar_dhawan/
│   ├── vijay_shankar/
│   ├── virat_kohli/
│   └── yuzvendra_chahal/
├── app.py
├── requirements.txt
└── README.md
🛠️ Tech Stack
Frontend & Deployment: Streamlit / Streamlit Community Cloud

Backend & Computer Vision: Python, OpenCV, face_recognition (dlib)

Data Processing: NumPy, Pillow, Scikit-Learn
Problem statement: Develop a machine learning/computer vision system that can identify a person from an image based on their facial features. The system should be trained using images of selected Indian cricket team members and classify a new input image into the corresponding person's name.
