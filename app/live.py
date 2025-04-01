import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from utils import load_model, identify_face, load_database, save_database
import streamlit as st


class FaceRecognizer(VideoTransformerBase):
    def __init__(self):
        self.current_name = ""
        self.current_box = None
        self.display_name = ""
        self.last_valid_embedding = None

    def transform(self, frame):
        face_app = load_model()
        img = frame.to_ndarray(format="bgr24")
        faces = face_app.get(img)

        if faces:
            face = faces[0]
            box = face.bbox.astype(int)
            name, dist = identify_face(face.embedding)

            self.current_name = name
            self.display_name = f"{name} ({dist:.2f})" if dist else name
            self.last_valid_embedding = face.embedding

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img, self.display_name, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return img


def live_recognition():
    st.header("🟢 Real-Time Facial Recognition via Webcam")

    ctx = webrtc_streamer(
        key="realtime",
        video_transformer_factory=FaceRecognizer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    placeholder = st.empty()

    if ctx and ctx.state.playing and ctx.video_transformer:
        detected_name = ctx.video_transformer.current_name
        
        if detected_name:
            placeholder.markdown(f"**👤 Detected Person:** {detected_name}")

        with st.expander("➕ Register this face"):
            name = st.text_input("Name to register:", key="live_name_input")
            if st.button("Save face") and name:
                embedding = ctx.video_transformer.last_valid_embedding

                if embedding is None:
                    st.error("❌ No valid face detected for registration.")
                    return

                if detected_name != "Unknown" and detected_name is not None:
                    if name != detected_name:
                        st.error(f"❌ Input name must match detected name: '{detected_name}'")
                        return

                database = load_database()
                database.setdefault(name, []).append(embedding)
                save_database(database)
                st.success(f"✅ Face successfully registered as '{name}'")
