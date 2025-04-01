import streamlit as st
st.set_page_config(page_title="Facial Recognition System", layout="centered")

from utils import load_model, identify_face
from recognition import register_face, recognize_image, display_database
from live import FaceRecognizer, live_recognition

st.title("🧠 Facial Recognition System with InsightFace")

tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Register Face", 
    "🔍 Recognize Image", 
    "📂 Database", 
    "🟢 Live Recognition"
])

with tab1:
    register_face()

with tab2:
    recognize_image()

with tab3:
    display_database()

with tab4:
    live_recognition()

