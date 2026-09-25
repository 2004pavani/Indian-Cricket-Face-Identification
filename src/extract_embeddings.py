from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis


# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(r"E:\Indian-Cricket-Face-Identification")

DATASET_DIR = PROJECT_DIR / "processed_dataset"

OUTPUT_DIR = PROJECT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "face_embeddings.csv"


# ==========================================
# LOAD INSIGHTFACE
# ==========================================

print("=" * 60)
print("LOADING INSIGHTFACE")
print("=" * 60)

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded successfully!\n")


# ==========================================
# IMAGE EXTENSIONS
# ==========================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ==========================================
# EXTRACT EMBEDDINGS
# ==========================================

records = []

total_images = 0
one_face_images = 0


player_folders = sorted(
    [folder for folder in DATASET_DIR.iterdir() if folder.is_dir()]
)


for player_folder in player_folders:

    print("=" * 60)
    print(f"PLAYER: {player_folder.name}")
    print("=" * 60)

    image_files = sorted(
        [
            file for file in player_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    player_count = 0

    for image_file in image_files:

        total_images += 1

        image = cv2.imread(str(image_file))

        if image is None:
            continue

        try:
            faces = app.get(image)
        except Exception as e:
            print(f"[ERROR] {image_file.name}: {e}")
            continue

        # We only use images containing exactly one face
        if len(faces) != 1:
            continue

        face = faces[0]

        # Get embedding
        embedding = face.embedding

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Store information
        record = {
            "player": player_folder.name,
            "image": image_file.name,
            "detection_score": float(face.det_score),
        }

        # Add 512 embedding values
        for i, value in enumerate(embedding):
            record[f"embedding_{i}"] = float(value)

        records.append(record)

        one_face_images += 1
        player_count += 1

    print(f"Embeddings extracted: {player_count}")


# ==========================================
# SAVE DATA
# ==========================================

df = pd.DataFrame(records)

df.to_csv(OUTPUT_FILE, index=False)


# ==========================================
# SUMMARY
# ==========================================

print("\n")
print("=" * 60)
print("EMBEDDING EXTRACTION COMPLETE")
print("=" * 60)

print(f"Total images scanned       : {total_images}")
print(f"One-face embeddings        : {one_face_images}")
print(f"Embedding dimensions       : 512")

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)