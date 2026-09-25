from pathlib import Path
import pandas as pd
import shutil


# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(r"E:\Indian-Cricket-Face-Identification")

COMBINED_DATASET = PROJECT_DIR / "combined_dataset"

MANIFEST_FILE = (
    PROJECT_DIR
    / "models"
    / "clean_dataset_manifest.csv"
)

PROCESSED_DATASET = (
    PROJECT_DIR
    / "processed_dataset"
)


# ==========================================
# LOAD MANIFEST
# ==========================================

print("=" * 60)
print("CREATING PROCESSED DATASET")
print("=" * 60)

manifest = pd.read_csv(MANIFEST_FILE)

keep_df = manifest[
    manifest["status"] == "keep"
].copy()

print(f"Images to copy: {len(keep_df)}")


# ==========================================
# CREATE PLAYER FOLDERS
# ==========================================

for player in sorted(keep_df["player"].unique()):

    folder = PROCESSED_DATASET / player

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================
# COPY APPROVED IMAGES
# ==========================================

copied = 0
missing = 0


for _, row in keep_df.iterrows():

    player = row["player"]
    image_name = row["image"]

    source = (
        COMBINED_DATASET
        / player
        / image_name
    )

    destination = (
        PROCESSED_DATASET
        / player
        / image_name
    )

    if not source.exists():

        print(f"[MISSING] {source}")
        missing += 1
        continue

    shutil.copy2(
        source,
        destination
    )

    copied += 1


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n")
print("=" * 60)
print("PROCESSED DATASET CREATED")
print("=" * 60)

print(f"Images requested : {len(keep_df)}")
print(f"Images copied    : {copied}")
print(f"Missing images   : {missing}")

print("\nPlayer-wise counts:")

for player in sorted(keep_df["player"].unique()):

    folder = PROCESSED_DATASET / player

    count = len([
        f for f in folder.iterdir()
        if f.is_file()
    ])

    print(f"{player:20} : {count}")

print("\nDataset location:")
print(PROCESSED_DATASET)

print("=" * 60)