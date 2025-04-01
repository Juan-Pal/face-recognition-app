# 🧠 Real-Time Face Recognition App

A real-time facial recognition system built with **Streamlit** and powered by **InsightFace** for high-accuracy face embeddings. This app allows users to register faces (via image or webcam), recognize them in real time, and manage a local facial database — all with a clean interface.

---

## 🚀 Features

- Register faces from uploaded images or webcam feed
- Real-time facial recognition via webcam
- Visual feedback with bounding boxes and identity labels
- Embeddings stored locally in a `.pkl` file
- Minimal and user-friendly Streamlit UI

---

## 📁 Project Structure

```
app/
├── main.py              # Streamlit UI main script
├── recognition.py       # Face registration and image recognition
├── live.py              # Webcam recognition with Streamlit WebRTC
├── utils.py             # Model loading & embedding management
├── dataloader.py        # CelebA embedding generator
data/
├── embeddings_celeba.pkl
├── identity_CelebA.txt
requirements.txt
README.md
```

---

## 🛠️ Installation

```bash
# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app/main.py
```

---

## 🧪 Generate Embeddings (Optional)

To generate your own face embeddings using the **CelebA** dataset:

```bash
python app/dataloader.py
```

Ensure you have:

- `img_align_celeba/` with aligned face images  
- `data/identity_CelebA.txt` with image-to-person mappings

---

## 📝 Dataset Attribution

This project utilizes the **CelebA** dataset for demonstration purposes.  
The dataset is available via **Kaggle** for easier access.

## 📂 Dataset Access

[![Kaggle Dataset](https://img.shields.io/badge/CelebA-Kaggle-blue?logo=kaggle)](https://www.kaggle.com/datasets/juanpal/images-of-famous-peoplec-celeba)


### 📚 Citation

If you use CelebA in a publication, please cite:

```
@inproceedings{liu2015faceattributes,
  author = {Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang},
  title = {Deep Learning Face Attributes in the Wild},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = December,
  year = {2015}
}
```

> ⚠️ **Note:** This dataset is intended strictly for non-commercial research and educational use.

---

## 📸 Example

Below is a screenshot of the face recognition system in action, identifying a face from an uploaded image:

![Demo](https://i.imgur.com/E8uq3SZ.png)
---

## 🧊 License

This project is intended for **educational and research purposes only**.
