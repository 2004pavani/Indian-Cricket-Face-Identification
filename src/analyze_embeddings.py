from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(r"E:\Indian-Cricket-Face-Identification")

EMBEDDINGS_FILE = (
    PROJECT_DIR / "models" / "face_embeddings.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "models" / "embedding_analysis.csv"
)


# ==========================================
# LOAD DATA
# ==========================================

print("=" * 60)
print("LOADING EMBEDDINGS")
print("=" * 60)

df = pd.read_csv(EMBEDDINGS_FILE)

print(f"Total embeddings: {len(df)}")


# ==========================================
# GET EMBEDDING COLUMNS
# ==========================================

embedding_columns = [
    column
    for column in df.columns
    if column.startswith("embedding_")
]

print(f"Embedding dimensions: {len(embedding_columns)}")


# ==========================================
# CONVERT TO NUMPY
# ==========================================

embeddings = df[embedding_columns].values.astype(np.float32)

players = sorted(df["player"].unique())

print("\nPlayers:")
for player in players:
    print(f" - {player}")


# ==========================================
# CALCULATE PLAYER CENTROIDS
# ==========================================

centroids = {}

for player in players:

    player_embeddings = embeddings[
        df["player"].values == player
    ]

    centroid = np.mean(player_embeddings, axis=0)

    # Normalize centroid
    centroid = centroid / np.linalg.norm(centroid)

    centroids[player] = centroid


# ==========================================
# COSINE SIMILARITY
# ==========================================

def cosine_similarity(vector1, vector2):

    denominator = (
        np.linalg.norm(vector1)
        * np.linalg.norm(vector2)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vector1, vector2) / denominator
    )


# ==========================================
# ANALYZE EVERY IMAGE
# ==========================================

results = []


for index, row in df.iterrows():

    embedding = embeddings[index]

    similarities = {}

    for player in players:

        similarities[player] = cosine_similarity(
            embedding,
            centroids[player]
        )

    # Player assigned by highest similarity
    predicted_player = max(
        similarities,
        key=similarities.get
    )

    own_player = row["player"]

    own_similarity = similarities[own_player]

    # Highest similarity among other players
    other_similarities = {
        player: score
        for player, score in similarities.items()
        if player != own_player
    }

    closest_other_player = max(
        other_similarities,
        key=other_similarities.get
    )

    closest_other_similarity = (
        other_similarities[closest_other_player]
    )

    margin = (
        own_similarity
        - closest_other_similarity
    )

    result = {
        "player": own_player,
        "image": row["image"],
        "detection_score": row["detection_score"],
        "own_similarity": own_similarity,
        "closest_other_player": closest_other_player,
        "closest_other_similarity": closest_other_similarity,
        "margin": margin,
        "predicted_player": predicted_player,
    }

    results.append(result)


# ==========================================
# CREATE ANALYSIS DATAFRAME
# ==========================================

analysis_df = pd.DataFrame(results)


# ==========================================
# SORT BY LOWEST MARGIN
# ==========================================

analysis_df = analysis_df.sort_values(
    by="margin",
    ascending=True
)


# ==========================================
# SAVE RESULTS
# ==========================================

analysis_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# DISPLAY MOST SUSPICIOUS
# ==========================================

print("\n")
print("=" * 60)
print("MOST SUSPICIOUS IMAGES")
print("=" * 60)

print(
    analysis_df[
        [
            "player",
            "image",
            "own_similarity",
            "closest_other_player",
            "closest_other_similarity",
            "margin",
            "predicted_player"
        ]
    ].head(20).to_string(index=False)
)


# ==========================================
# SUMMARY
# ==========================================

print("\n")
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print(f"Total images analyzed : {len(analysis_df)}")

print("\nPredicted player matches:")

matches = (
    analysis_df["player"]
    == analysis_df["predicted_player"]
)

print(f"Matches     : {matches.sum()}")
print(f"Different   : {(~matches).sum()}")

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)