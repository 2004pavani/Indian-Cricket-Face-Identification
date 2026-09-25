from pathlib import Path
import pandas as pd
import shutil


# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(r"E:\Indian-Cricket-Face-Identification")

COMBINED_DATASET = PROJECT_DIR / "combined_dataset"

ANALYSIS_FILE = (
    PROJECT_DIR / "models" / "embedding_analysis.csv"
)

REVIEW_DIR = (
    PROJECT_DIR / "review_candidates"
)


# ==========================================
# LOAD ANALYSIS
# ==========================================

df = pd.read_csv(ANALYSIS_FILE)

# Sort by smallest margin
df = df.sort_values(
    by="margin",
    ascending=True
)


# ==========================================
# TAKE TOP 20 SUSPICIOUS IMAGES
# ==========================================

review_df = df.head(20).copy()


# ==========================================
# CREATE REVIEW FOLDER
# ==========================================

REVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# COPY IMAGES
# ==========================================

for index, row in review_df.iterrows():

    player = row["player"]
    image_name = row["image"]

    source = (
        COMBINED_DATASET
        / player
        / image_name
    )

    if not source.exists():
        print(f"[NOT FOUND] {source}")
        continue

    # Add ranking to filename
    rank = len(list(REVIEW_DIR.iterdir())) + 1

    new_name = (
        f"{rank:02d}__"
        f"{player}__"
        f"{image_name}"
    )

    destination = REVIEW_DIR / new_name

    shutil.copy2(
        source,
        destination
    )


# ==========================================
# SAVE REPORT
# ==========================================

report_file = REVIEW_DIR / "review_report.csv"

review_df.to_csv(
    report_file,
    index=False
)


# ==========================================
# OUTPUT
# ==========================================

print("=" * 60)
print("REVIEW SET CREATED")
print("=" * 60)

print(f"Images copied : {len(review_df)}")

print(f"\nReview folder:")
print(REVIEW_DIR)

print(f"\nReport:")
print(report_file)

print("=" * 60)

print("\nTop candidates:")
print(
    review_df[
        [
            "player",
            "image",
            "own_similarity",
            "closest_other_player",
            "closest_other_similarity",
            "margin",
            "predicted_player"
        ]
    ].to_string(index=False)
)