import pickle
import os
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import streamlit as st

DB_FILE = "embeddings_celeba.pkl"

@st.cache_resource
def load_model():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)
    return app

def identify_face(new_embedding, threshold=0.4):
    database = load_database()
    best_match = None
    smallest_distance = float("inf")

    for name, embeddings in database.items():
        for emb in embeddings:
            dist = cosine(new_embedding, emb)
            if dist < smallest_distance:
                smallest_distance = dist
                best_match = name

    if smallest_distance < threshold:
        return best_match, smallest_distance
    return "Unknown", None

def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_database(database):
    with open(DB_FILE, "wb") as f:
        pickle.dump(database, f)
