import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "clean_dataset_manifest.csv"
)

EMBEDDING_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "face_embeddings.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING EMBEDDINGS")
print("=" * 60)

manifest = pd.read_csv(MANIFEST_PATH)

# Only approved images
manifest = manifest[
    manifest["status"] == "keep"
].copy()

embeddings = pd.read_csv(EMBEDDING_PATH)

embedding_columns = [
    column
    for column in embeddings.columns
    if column.startswith("embedding_")
]

# Merge labels with embeddings
data = manifest.merge(
    embeddings,
    on=["player", "image"],
    how="inner"
)

print(f"Total images: {len(data)}")
print(f"Embedding dimensions: {len(embedding_columns)}")


# ============================================================
# SPLIT BY SOURCE
# ============================================================

data["source"] = data["image"].apply(
    lambda x: "kaggle"
    if str(x).startswith("kaggle_")
    else "github"
)

train_data = data[
    data["source"] == "kaggle"
].copy()

test_data = data[
    data["source"] == "github"
].copy()

print("\n" + "=" * 60)
print("CROSS-SOURCE COSINE SIMILARITY")
print("=" * 60)

print(
    f"\nTraining images (Kaggle): "
    f"{len(train_data)}"
)

print(
    f"Testing images (GitHub): "
    f"{len(test_data)}"
)


# ============================================================
# PREPARE TRAINING EMBEDDINGS
# ============================================================

X_train = train_data[
    embedding_columns
].to_numpy(dtype=np.float32)

y_train = train_data[
    "player"
].to_numpy(dtype=str)


# Normalize training embeddings

train_norms = np.linalg.norm(
    X_train,
    axis=1,
    keepdims=True
)

X_train = X_train / np.maximum(
    train_norms,
    1e-12
)


# ============================================================
# CREATE CENTROIDS FROM KAGGLE ONLY
# ============================================================

players = sorted(
    train_data["player"].unique()
)

centroids = {}

print("\n" + "=" * 60)
print("CREATING KAGGLE PLAYER CENTROIDS")
print("=" * 60)

for player in players:

    player_embeddings = X_train[
        y_train == player
    ]

    centroid = np.mean(
        player_embeddings,
        axis=0
    )

    # Normalize centroid
    centroid = centroid / np.maximum(
        np.linalg.norm(centroid),
        1e-12
    )

    centroids[player] = centroid

    print(
        f"{player}: "
        f"{len(player_embeddings)} training images"
    )


# ============================================================
# PREPARE GITHUB TEST EMBEDDINGS
# ============================================================

X_test = test_data[
    embedding_columns
].to_numpy(dtype=np.float32)

y_test = test_data[
    "player"
].to_numpy(dtype=str)


test_norms = np.linalg.norm(
    X_test,
    axis=1,
    keepdims=True
)

X_test = X_test / np.maximum(
    test_norms,
    1e-12
)


# ============================================================
# CALCULATE COSINE SIMILARITY
# ============================================================

centroid_matrix = np.vstack(
    [
        centroids[player]
        for player in players
    ]
)

similarity_matrix = cosine_similarity(
    X_test,
    centroid_matrix
)

predicted_indices = np.argmax(
    similarity_matrix,
    axis=1
)

y_pred = np.array(
    [
        players[index]
        for index in predicted_indices
    ]
)


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CROSS-SOURCE RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy percentage: "
    f"{accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=players
)

print("\nConfusion Matrix:")

print("\nPlayers:")
print(players)

print("\nMatrix:")
print(cm)


# ============================================================
# WRONG PREDICTIONS
# ============================================================

results = test_data[
    ["player", "image"]
].copy()

results["predicted"] = y_pred

wrong = results[
    results["player"] != results["predicted"]
]

print("\n" + "=" * 60)
print("WRONG PREDICTIONS")
print("=" * 60)

if len(wrong) == 0:
    print("No wrong predictions.")
else:
    print(
        wrong.to_string(index=False)
    )


print("\nTotal test images:", len(test_data))
print(
    "Correct:",
    (y_test == y_pred).sum()
)

print(
    "Wrong:",
    (y_test != y_pred).sum()
)