import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from insightface.app import FaceAnalysis


# ==============================
# PROJECT PATHS
# ==============================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "clean_dataset_manifest.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "svm_face_classifier.pkl"
)

EMBEDDING_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "face_embeddings.csv"
)


# ==============================
# LOAD CLEAN MANIFEST
# ==============================

manifest = pd.read_csv(MANIFEST_PATH)

# Keep only approved images
manifest = manifest[manifest["status"] == "keep"].copy()

# Identify source from filename
manifest["source"] = manifest["image"].apply(
    lambda x: "kaggle" if str(x).startswith("kaggle_") else "github"
)

print("\n===================================")
print("CROSS-SOURCE EVALUATION")
print("===================================")

print("\nImages by source:")
print(manifest["source"].value_counts())

print("\nImages by player and source:")
print(
    pd.crosstab(
        manifest["player"],
        manifest["source"]
    )
)


# ==============================
# LOAD EMBEDDINGS
# ==============================

embeddings = pd.read_csv(EMBEDDING_PATH)

embedding_columns = [
    col for col in embeddings.columns
    if col.startswith("embedding_")
]

# Merge manifest with embeddings
data = manifest.merge(
    embeddings,
    on=["player", "image"],
    how="inner"
)

print("\nTotal usable records:", len(data))


# ==============================
# KAGGLE → TRAIN
# GITHUB → TEST
# ==============================

train_data = data[data["source"] == "kaggle"].copy()
test_data = data[data["source"] == "github"].copy()

print("\nTraining images (Kaggle):", len(train_data))
print("Testing images (GitHub):", len(test_data))


# ==============================
# PREPARE DATA
# ==============================

X_train = train_data[embedding_columns].to_numpy(
    dtype=np.float32
)

y_train = train_data["player"].to_numpy(dtype=str)

X_test = test_data[embedding_columns].to_numpy(
    dtype=np.float32
)

y_test = test_data["player"].to_numpy(dtype=str)


# ==============================
# TRAIN A NEW SVM
# ==============================

from sklearn.svm import SVC

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)


# ==============================
# PREDICTION
# ==============================

y_pred = model.predict(X_test)


# ==============================
# RESULTS
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("RESULTS")
print("===================================")

print(f"\nCross-source accuracy: {accuracy:.4f}")
print(f"Cross-source accuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

labels = sorted(data["player"].unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print("\nPlayers:")
print(labels)

print("\nMatrix:")
print(cm)


# ==============================
# WRONG PREDICTIONS
# ==============================

results = test_data[
    ["player", "image"]
].copy()

results["predicted"] = y_pred

wrong = results[
    results["player"] != results["predicted"]
]

print("\n===================================")
print("WRONG PREDICTIONS")
print("===================================")

if len(wrong) == 0:
    print("No wrong predictions.")
else:
    print(wrong.to_string(index=False))

print("\nTotal test images:", len(test_data))
print("Correct:", (y_test == y_pred).sum())
print("Wrong:", (y_test != y_pred).sum())