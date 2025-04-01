import os
import cv2
import pickle
from collections import defaultdict
from insightface.app import FaceAnalysis

def generate_embeddings_from_folder(
    image_folder="img_align_celeba",
    identity_file="identity_CelebA.txt",
    celebrity_limit=50,
    images_per_celebrity=15,
    output_file="embeddings_celeba.pkl"
):
    app = FaceAnalysis()
    app.prepare(ctx_id=0)  # 0 = GPU, -1 = CPU

    identities = defaultdict(list)

    with open(identity_file, "r") as f:
        for line in f:
            image_name, person_id = line.strip().split()
            identities[person_id].append(image_name)

    selected_identities = list(identities.items())[:celebrity_limit]
    embeddings_dict = {}

    for person_id, images in selected_identities:
        print(f"🔍 Processing ID {person_id}...")
        embeddings_dict[f"id_{person_id}"] = []

        for image_name in images[:images_per_celebrity]:
            img_path = os.path.join(image_folder, image_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"❌ Failed to load: {img_path}")
                continue

            faces = app.get(img)
            if not faces:
                print(f"⚠️ {image_name} – no faces detected.")
                continue

            embedding = faces[0].embedding
            embeddings_dict[f"id_{person_id}"].append(embedding)

    with open(output_file, "wb") as f:
        pickle.dump(embeddings_dict, f)

    print(f"✅ Embeddings saved to '{output_file}'")
