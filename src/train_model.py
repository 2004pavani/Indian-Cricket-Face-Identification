from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# PATHS
# ==========================================

PROJECT_DIR = Path(
    r"E:\Indian-Cricket-Face-Identification"
)

EMBEDDINGS_FILE = (
    PROJECT_DIR
    / "models"
    / "face_embeddings.csv"
)

MANIFEST_FILE = (
    PROJECT_DIR
    / "models"
    / "clean_dataset_manifest.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

MODEL_FILE = MODEL_DIR / "svm_face_classifier.pkl"


# ==========================================
# LOAD DATA
# ==========================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

embeddings_df = pd.read_csv(
    EMBEDDINGS_FILE
)

manifest_df = pd.read_csv(
    MANIFEST_FILE
)


# ==========================================
# KEEP ONLY APPROVED IMAGES
# ==========================================

approved = manifest_df[
    manifest_df["status"] == "keep"
][
    ["player", "image"]
]


# Merge embeddings with approved images
df = embeddings_df.merge(
    approved,
    on=["player", "image"],
    how="inner"
)


print(f"Approved images: {len(df)}")


# ==========================================
# EXTRACT FEATURES
# ==========================================

embedding_columns = [
    column
    for column in df.columns
    if column.startswith("embedding_")
]

X = df[embedding_columns].to_numpy(
    dtype=np.float32
)

y = df["player"].to_numpy(
    dtype=str
)


print(f"Features: {X.shape}")
print(f"Classes: {sorted(np.unique(y))}")


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nDataset split:")
print(f"Training images: {len(X_train)}")
print(f"Testing images : {len(X_test)}")


# ==========================================
# TRAIN SVM
# ==========================================

print("\n" + "=" * 60)
print("TRAINING SVM")
print("=" * 60)

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(
    X_train,
    y_train
)

print("SVM training completed!")


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(
    X_test
)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy percentage: "
    f"{accuracy * 100:.2f}%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

labels = sorted(
    np.unique(y)
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


print("\nConfusion Matrix:")

print(
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_FILE
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(MODEL_FILE)

print("=" * 60)