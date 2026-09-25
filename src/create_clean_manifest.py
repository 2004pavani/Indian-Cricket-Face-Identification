from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(r"E:\Indian-Cricket-Face-Identification")

EMBEDDINGS_FILE = (
    PROJECT_DIR / "models" / "face_embeddings.csv"
)

ANALYSIS_FILE = (
    PROJECT_DIR / "models" / "dataset_quality_report.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "models" / "clean_dataset_manifest.csv"
)


# Load data
embeddings_df = pd.read_csv(EMBEDDINGS_FILE)
analysis_df = pd.read_csv(ANALYSIS_FILE)


# Confirmed wrong images identified during our review
confirmed_wrong = {
    ("Hardik_Pandya", "github_804083b27f.jpg"),
    ("MS_Dhoni", "kaggle_Image_17.jpg"),
    ("MS_Dhoni", "kaggle_Image_16.jpg"),
}


records = []


for _, row in embeddings_df.iterrows():

    player = row["player"]
    image = row["image"]

    key = (player, image)

    # Default: keep
    status = "keep"
    reason = "Exactly one face detected"

    # Remove the three confirmed wrong labels
    if key in confirmed_wrong:

        status = "exclude"
        reason = "Confirmed incorrect player label"

    records.append({
        "player": player,
        "image": image,
        "detection_score": row["detection_score"],
        "status": status,
        "reason": reason
    })


manifest = pd.DataFrame(records)


# Save manifest
manifest.to_csv(
    OUTPUT_FILE,
    index=False
)


# Summary
print("=" * 60)
print("CLEAN DATASET MANIFEST")
print("=" * 60)

print(f"Total one-face images : {len(manifest)}")

print(
    f"Keep                  : "
    f"{(manifest['status'] == 'keep').sum()}"
)

print(
    f"Exclude               : "
    f"{(manifest['status'] == 'exclude').sum()}"
)

print("\nPer-player counts:")

print(
    manifest[
        manifest["status"] == "keep"
    ]
    .groupby("player")
    .size()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)