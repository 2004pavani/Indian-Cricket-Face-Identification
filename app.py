import os
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "svm_face_classifier.pkl"
)

EMBEDDINGS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_embeddings.csv"
)

UNKNOWN_THRESHOLD = 0.50


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian Cricket Face Identifier",
    page_icon="🏏",
    layout="centered"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main project heading */
    .main-title {
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        line-height: 1.15;
        margin-top: 5px;
        margin-bottom: 8px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 18px;
        margin-bottom: 32px;
    }

    /* Section headings */
    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 15px;
    }

    /* Identification result */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        background-color: #e8f7ee;
        border: 1px solid #b7e4c7;
        margin-top: 20px;
    }

    .player-name {
        text-align: center;
        font-size: 34px;
        font-weight: 800;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    /* Unknown result */
    .unknown-box {
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        background-color: #fff3cd;
        border: 1px solid #ffe69c;
        margin-top: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777777;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 15px;
        border-top: 1px solid #dddddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD SVM MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"SVM model not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

@st.cache_resource
def load_face_model():

    face_model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    face_model.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    return face_model


# ============================================================
# LOAD PLAYER CENTROIDS
# ============================================================

@st.cache_resource
def load_player_centroids():

    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"Embeddings file not found:\n{EMBEDDINGS_PATH}"
        )

    df = pd.read_csv(EMBEDDINGS_PATH)

    embedding_columns = [
        col for col in df.columns
        if col.startswith("embedding_")
    ]

    centroids = {}

    for player, group in df.groupby("player"):

        embeddings = group[
            embedding_columns
        ].to_numpy(dtype=np.float32)

        centroid = np.mean(
            embeddings,
            axis=0
        )

        norm = np.linalg.norm(centroid)

        if norm > 0:
            centroid = centroid / norm

        centroids[player] = centroid

    return centroids


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    svm_model = load_model()
    face_model = load_face_model()
    player_centroids = load_player_centroids()

except Exception as e:

    st.error(
        f"Model loading failed:\n\n{e}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏏 Indian Cricket Face Identifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered facial recognition system for selected Indian cricket players'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🏏 Project Information")

    st.write(
        "This system identifies selected Indian cricket "
        "players from an uploaded image."
    )

    st.divider()

    st.subheader("Supported Players")

    for player in sorted(player_centroids.keys()):

        display_name = player.replace("_", " ")

        st.write(f"• {display_name}")

    st.divider()

    st.subheader("Technology")

    st.write("🔹 InsightFace")
    st.write("🔹 ArcFace Embeddings")
    st.write("🔹 SVM Classifier")
    st.write("🔹 Cosine Similarity")
    st.write("🔹 Streamlit")

    st.divider()

    st.caption(
        "Unknown threshold: 0.50"
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📷 Upload Player Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing one person's face.",
    width="stretch"
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "Unable to read the uploaded image."
        )

        st.stop()


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    faces = face_model.get(image)


    # ========================================================
    # NO FACE
    # ========================================================

    if len(faces) == 0:

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            width="stretch"
        )

        st.error(
            "❌ No face detected. Please upload an image "
            "where the face is clearly visible."
        )

        st.stop()


    # ========================================================
    # MULTIPLE FACES
    # ========================================================

    if len(faces) > 1:

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            width="stretch"
        )

        st.warning(
            f"⚠️ {len(faces)} faces detected. "
            "Please upload an image containing only one person."
        )

        st.stop()


    # ========================================================
    # EXACTLY ONE FACE
    # ========================================================

    face = faces[0]

    detection_score = float(
        face.det_score
    )


    # --------------------------------------------------------
    # GENERATE EMBEDDING
    # --------------------------------------------------------

    embedding = face.embedding

    embedding = embedding / (
        np.linalg.norm(embedding) + 1e-10
    )

    embedding_input = embedding.reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # SVM PREDICTION
    # --------------------------------------------------------

    predicted_player = svm_model.predict(
        embedding_input
    )[0]


    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    similarity_scores = {}

    for player, centroid in player_centroids.items():

        score = cosine_similarity(
            embedding.reshape(1, -1),
            centroid.reshape(1, -1)
        )[0][0]

        similarity_scores[player] = float(score)


    best_player = max(
        similarity_scores,
        key=similarity_scores.get
    )

    best_similarity = similarity_scores[
        best_player
    ]


    # ========================================================
    # MAIN RESULT LAYOUT
    # ========================================================

    st.success(
        "Image uploaded successfully!"
    )

    col1, col2 = st.columns(
        [1.15, 1],
        gap="large"
    )


    # ========================================================
    # LEFT: IMAGE
    # ========================================================

    with col1:

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            width="stretch"
        )


    # ========================================================
    # RIGHT: IDENTIFICATION
    # ========================================================

    with col2:

        st.markdown(
            "### 🔎 Identification Result"
        )

        if best_similarity < UNKNOWN_THRESHOLD:

            st.warning(
                "⚠️ Unknown Person"
            )

            st.write(
                f"**Similarity:** "
                f"{best_similarity:.4f}"
            )

            st.write(
                "The face does not sufficiently match "
                "any player in the dataset."
            )

        else:

            display_name = predicted_player.replace(
                "_",
                " "
            )

            st.success(
                "✅ Player Identified"
            )

            st.markdown(
                f'<div class="player-name">'
                f'🏏 {display_name}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.write(
                f"**Similarity:** "
                f"{best_similarity:.4f}"
            )

            if predicted_player == best_player:

                st.success(
                    "SVM and cosine similarity agree."
                )

            else:

                st.warning(
                    "SVM and cosine similarity differ."
                )


    # ========================================================
    # METRICS
    # ========================================================

    st.markdown(
        "### 📊 Detection & Embedding Details"
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Faces Detected",
        len(faces),
        border=True
    )

    metric2.metric(
        "Detection Score",
        f"{detection_score:.4f}",
        border=True
    )

    metric3.metric(
        "Embedding",
        "512-D",
        border=True
    )


    # ========================================================
    # SIMILARITY DETAILS
    # ========================================================

    if best_similarity >= UNKNOWN_THRESHOLD:

        st.markdown(
            "### 🎯 Matching Details"
        )

        match_col1, match_col2 = st.columns(2)

        with match_col1:

            st.metric(
                "Identified Player",
                predicted_player.replace("_", " "),
                border=True
            )

        with match_col2:

            st.metric(
                "Cosine Similarity",
                f"{best_similarity:.4f}",
                border=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Indian Cricket Team Member Face Identification
        <br>
        Computer Vision • Face Embeddings • Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)