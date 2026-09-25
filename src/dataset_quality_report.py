from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_DIR = Path(r"E:\Indian-Cricket-Face-Identification")

EMBEDDINGS_FILE = (
    PROJECT_DIR / "models" / "face_embeddings.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "models" / "dataset_quality_report.csv"
)


# ==========================================
# LOAD EMBEDDINGS
# ==========================================

df = pd.read_csv(EMBEDDINGS_FILE)

embedding_columns = [
    c for c in df.columns
    if c.startswith("embedding_")
]

embeddings = df[embedding_columns].values.astype(np.float32)

players = sorted(df["player"].unique())


# ==========================================
# CALCULATE PLAYER CENTROIDS
# ==========================================

centroids = {}

for player in players:

    mask = df["player"].values == player

    player_embeddings = embeddings[mask]

    centroid = np.mean(player_embeddings, axis=0)

    norm = np.linalg.norm(centroid)

    if norm > 0:
        centroid = centroid / norm

    centroids[player] = centroid


# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ==========================================
# ANALYZE EACH IMAGE
# ==========================================

results = []

for i, row in df.iterrows():

    embedding = embeddings[i]

    similarities = {}

    for player in players:

        similarities[player] = cosine_similarity(
            embedding,
            centroids[player]
        )

    predicted_player = max(
        similarities,
        key=similarities.get
    )

    own_player = row["player"]

    own_similarity = similarities[own_player]

    other_scores = {
        p: s
        for p, s in similarities.items()
        if p != own_player
    }

    closest_other_player = max(
        other_scores,
        key=other_scores.get
    )

    closest_other_similarity = (
        other_scores[closest_other_player]
    )

    margin = (
        own_similarity -
        closest_other_similarity
    )

    results.append({
        "player": own_player,
        "image": row["image"],
        "detection_score": row["detection_score"],
        "own_similarity": own_similarity,
        "closest_other_player": closest_other_player,
        "closest_other_similarity": closest_other_similarity,
        "margin": margin,
        "predicted_player": predicted_player,
        "prediction_matches_label": (
            predicted_player == own_player
        )
    })


result_df = pd.DataFrame(results)


# ==========================================
# SAVE COMPLETE REPORT
# ==========================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# PRINT PLAYER SUMMARY
# ==========================================

print("=" * 70)
print("DATASET QUALITY REPORT")
print("=" * 70)

for player in players:

    player_df = result_df[
        result_df["player"] == player
    ]

    total = len(player_df)

    matches = player_df[
        player_df["prediction_matches_label"]
    ]

    mismatches = player_df[
        ~player_df["prediction_matches_label"]
    ]

    print(f"\n{player}")
    print("-" * 50)

    print(f"Total images       : {total}")
    print(f"Predicted same     : {len(matches)}")
    print(f"Predicted different: {len(mismatches)}")

    print(
        f"Average similarity : "
        f"{player_df['own_similarity'].mean():.4f}"
    )

    print(
        f"Minimum similarity : "
        f"{player_df['own_similarity'].min():.4f}"
    )

    print(
        f"Average margin     : "
        f"{player_df['margin'].mean():.4f}"
    )


# ==========================================
# OVERALL SUMMARY
# ==========================================

print("\n")
print("=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)

print(f"Total embeddings : {len(result_df)}")

print(
    f"Label matches    : "
    f"{result_df['prediction_matches_label'].sum()}"
)

print(
    f"Label mismatches : "
    f"{(~result_df['prediction_matches_label']).sum()}"
)

print(
    f"Match percentage : "
    f"{result_df['prediction_matches_label'].mean() * 100:.2f}%"
)

print("\nSaved report:")
print(OUTPUT_FILE)

print("=" * 70)