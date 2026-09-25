from pathlib import Path
import cv2
from insightface.app import FaceAnalysis


# ==========================================
# PATH
# ==========================================

PROJECT_DIR = Path(r"E:\Indian-Cricket-Face-Identification")

DATASET_DIR = PROJECT_DIR / "combined_dataset"


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
# SUPPORTED IMAGE TYPES
# ==========================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ==========================================
# DATASET ANALYSIS
# ==========================================

grand_total = 0
grand_one_face = 0
grand_no_face = 0
grand_multiple_faces = 0


player_folders = sorted(
    [folder for folder in DATASET_DIR.iterdir() if folder.is_dir()]
)


for player_folder in player_folders:

    total = 0
    one_face = 0
    no_face = 0
    multiple_faces = 0

    image_files = sorted(
        [
            file for file in player_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    print("=" * 60)
    print(f"PLAYER: {player_folder.name}")
    print("=" * 60)

    for image_file in image_files:

        total += 1

        image = cv2.imread(str(image_file))

        if image is None:
            print(f"[ERROR] Could not read: {image_file.name}")
            continue

        try:
            faces = app.get(image)
        except Exception as e:
            print(f"[ERROR] {image_file.name}: {e}")
            continue

        face_count = len(faces)

        if face_count == 0:
            no_face += 1

        elif face_count == 1:
            one_face += 1

        else:
            multiple_faces += 1

    print(f"Total images       : {total}")
    print(f"Exactly one face   : {one_face}")
    print(f"No face detected   : {no_face}")
    print(f"Multiple faces     : {multiple_faces}")

    grand_total += total
    grand_one_face += one_face
    grand_no_face += no_face
    grand_multiple_faces += multiple_faces


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n")
print("=" * 60)
print("FINAL DATASET ANALYSIS")
print("=" * 60)

print(f"Total images       : {grand_total}")
print(f"Exactly one face   : {grand_one_face}")
print(f"No face detected   : {grand_no_face}")
print(f"Multiple faces     : {grand_multiple_faces}")

print("=" * 60)