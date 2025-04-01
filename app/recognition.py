import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import load_model, identify_face, load_database, save_database


def register_face():
    st.header("Register a New Face")
    name = st.text_input("Name")
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    database = load_database()
    face_app = load_model()

    if name and file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        faces = face_app.get(img)

        if faces:
            embedding = faces[0].embedding
            database.setdefault(name, []).append(embedding)
            save_database(database)
            st.success(f"✅ Face registered as '{name}'.")
        else:
            st.error("❌ No face detected.")


def recognize_image():
    st.header("Recognize Face in Image")
    threshold = st.slider("Matching Threshold", 0.3, 1.0, 0.4, 0.01)
    file = st.file_uploader("Upload an image for recognition", type=["jpg", "jpeg", "png"], key="rec")
    face_app = load_model()

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        faces = face_app.get(img)

        if not faces:
            st.error("❌ No faces detected.")
        else:
            for face in faces:
                box = face.bbox.astype(int)
                name, dist = identify_face(face.embedding, threshold)
                label = f"{name} ({dist:.2f})" if dist else name
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img, label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Result")


def display_database():
    st.header("Registered Faces Database")
    database = load_database()
    st.write(f"👥 Total registered individuals: {len(database)}")
    st.json({k: len(v) for k, v in database.items()})

    if st.button("🗑️ Reset Database"):
        database.clear()
        save_database(database)
        st.success("✅ Database successfully reset.")
