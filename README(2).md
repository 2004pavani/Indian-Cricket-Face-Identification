# 🏏 Indian Cricket Team Member Face Identification

An AI-powered computer vision application that identifies selected Indian cricket team members from an uploaded face image.

The system uses **InsightFace** for face detection and 512-dimensional face embeddings, an **SVM classifier** for player classification, and **cosine similarity** for identity matching and unknown-person rejection. A Streamlit web application provides the user interface.

---

## 📌 Project Overview

The objective of this project is to develop a facial identification system that can recognize a person from an image and classify the person into one of the selected Indian cricket players used in the project.

### Supported Players

1. Virat Kohli
2. Rohit Sharma
3. MS Dhoni
4. Jasprit Bumrah
5. Hardik Pandya

The application accepts an image, detects the face, generates a 512-dimensional facial embedding, predicts the player using an SVM classifier, and compares the embedding with player centroids using cosine similarity.

If the similarity is below the configured threshold, the application rejects the person as an unknown individual.

---

## 🎯 Objectives

- Detect a face from an input image.
- Ensure that the input contains exactly one detectable face.
- Generate a numerical facial representation using InsightFace.
- Convert each face into a 512-dimensional embedding.
- Train an SVM classifier to identify the selected cricket players.
- Use cosine similarity to compare an input face with player centroids.
- Reject faces that do not sufficiently match the known player dataset.
- Provide an easy-to-use Streamlit web application.
- Evaluate the system using held-out, cross-source, and external test images.

---

## 🧠 System Architecture

```text
                    Input Image
                         │
                         ▼
                 Face Detection
                 using InsightFace
                         │
                         ▼
                  Single Face Check
                    /           \
                  No             Yes
                  │               │
                  ▼               ▼
               Reject       512-D Face
                              Embedding
                                  │
                                  ▼
                           SVM Classifier
                                  │
                                  ▼
                         Player Prediction
                                  │
                                  ▼
                         Cosine Similarity
                         with Player Centroids
                                  │
                     ┌────────────┴────────────┐
                     │                         │
              Similarity >= 0.50       Similarity < 0.50
                     │                         │
                     ▼                         ▼
              Known Player              Unknown Person
```

---

## 🔄 Project Workflow

The complete development workflow was:

```text
Raw Datasets
     ↓
Dataset Preparation
     ↓
Combined Dataset
     ↓
Face Detection Analysis
     ↓
Dataset Cleaning
     ↓
Processed Dataset
     ↓
Face Embedding Extraction
     ↓
Embedding Analysis
     ↓
SVM Training
     ↓
Cross-Source Evaluation
     ↓
Cosine Similarity Evaluation
     ↓
Streamlit Application
```

---

## 📂 Project Structure

```text
Indian-Cricket-Face-Identification/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── face_embeddings.csv
│   └── svm_face_classifier.pkl
│
└── src/
    ├── analyze_dataset.py
    ├── analyze_embeddings.py
    ├── cosine_similarity_test.py
    ├── create_clean_manifest.py
    ├── create_processed_dataset.py
    ├── create_review_set.py
    ├── cross_source_test.py
    ├── dataset_quality_report.py
    ├── extract_embeddings.py
    ├── prepare_dataset.py
    └── train_model.py
```

### Local Development Files

During development, additional raw datasets, intermediate datasets, review images, and testing scripts were used locally. These are intentionally not required by the deployed Streamlit application.

The final application requires the trained model and embedding data stored in `models/`.

---

## 📊 Dataset

Two image sources were combined during development:

### Dataset 1 — Kaggle Indian Cricketers Dataset

For the five selected players, the original dataset contained:

- Virat Kohli — 25 images
- Rohit Sharma — 25 images
- MS Dhoni — 25 images
- Jasprit Bumrah — 25 images
- Hardik Pandya — 25 images

Total: **125 images**

### Dataset 2 — GitHub Cricket Face Dataset

The second source contained:

- Virat Kohli — 48 images
- Rohit Sharma — 50 images
- MS Dhoni — 46 images
- Jasprit Bumrah — 21 images
- Hardik Pandya — 40 images

Total: **205 images**

### Combined Dataset

The two sources produced:

**330 total images**

After InsightFace-based face analysis and dataset cleaning, the final processed dataset used for embedding extraction contained:

- Hardik Pandya — 55
- Jasprit Bumrah — 38
- MS Dhoni — 57
- Rohit Sharma — 66
- Virat Kohli — 60

Total usable images: **276**

Two additional manually identified mislabeled Rohit Sharma images were removed from the processed dataset, resulting in **274 images used for the final embeddings/model evaluation**.

---

## 👁️ Face Detection

The project uses **InsightFace** with the `buffalo_l` model pack.

For every image, the system checks the number of detected faces.

### Detection rules

```text
0 faces
   → Reject image

More than 1 face
   → Ask user to upload an image containing one person

Exactly 1 face
   → Continue identification
```

This prevents the system from attempting to identify an image containing multiple people.

---

## 🔢 Face Embeddings

After detecting exactly one face, InsightFace generates a numerical representation of the face.

Each face is represented using a:

**512-dimensional embedding**

The embedding captures facial characteristics in numerical form and is used as the input feature vector for classification and similarity comparison.

The embeddings are normalized before classification and similarity calculations.

---

## 🤖 SVM Classifier

The primary classification model used by the application is a:

**Support Vector Machine (SVM)**

Configuration used during final training:

```text
Kernel: RBF
C: 10
Gamma: scale
Test size: 20%
Random state: 42
Stratified split: Yes
```

The SVM receives the 512-dimensional face embedding and predicts one of the five supported players.

---

## 📐 Cosine Similarity

Cosine similarity is used as a supporting identity-matching mechanism.

For each player, a centroid is calculated from the available face embeddings.

The uploaded face embedding is then compared with each player's centroid.

The player with the highest cosine similarity is considered the closest match.

### Unknown Person Rejection

The application uses:

```text
UNKNOWN_THRESHOLD = 0.50
```

If the highest similarity is below `0.50`, the application reports:

> Unknown Person

The threshold is a project-level heuristic and was retained based on the observed separation in the available known-player and unknown-person tests. It is not claimed to be a universally validated face-recognition threshold.

---

## 🧪 Model Evaluation

Several evaluation approaches were performed.

### 1. Random Held-Out SVM Test

The dataset was divided into training and testing portions using a stratified 80/20 split.

Results:

```text
Training images: 219
Test images: 55
Accuracy: 100%
```

All 55 held-out test images were classified correctly.

This result applies to this particular held-out split and should not be interpreted as universal real-world accuracy.

---

### 2. Cross-Source SVM Evaluation

A stronger source-separated evaluation was performed.

```text
Kaggle images → Training
GitHub images → Testing
```

Final evaluation:

```text
Training images: 116
Testing images: 158

Accuracy: 100%
Correct: 158 / 158
Incorrect: 0
```

Precision, recall, and F1-score were 1.00 for each of the five classes in this evaluation.

---

### 3. Cross-Source Cosine Similarity Evaluation

The same source-separated setup was used for cosine similarity:

```text
Kaggle images → Player centroids
GitHub images → Test images
```

Results:

```text
Testing images: 158
Correct: 158
Incorrect: 0
Accuracy: 100%
```

Precision, recall and F1-score were 1.00 for each class in this evaluation.

---

## 🌐 External Image Testing

After model evaluation, five external images from outside the training datasets were tested through the Streamlit application.

| Player | Cosine Similarity | Result |
|---|---:|---|
| Rohit Sharma | 0.8025 | ✅ Correct |
| Virat Kohli | 0.8613 | ✅ Correct |
| MS Dhoni | 0.8882 | ✅ Correct |
| Jasprit Bumrah | 0.7685 | ✅ Correct |
| Hardik Pandya | 0.7649 | ✅ Correct |

External test result:

```text
5 / 5 correctly identified
```

These five tests are an additional small external validation sample and are not presented as proof of universal real-world accuracy.

---

## 🚫 Unknown Person Test

An unrelated person's image was also tested.

Observed result:

```text
Cosine similarity: 0.1455
Unknown threshold: 0.50
```

Since:

```text
0.1455 < 0.50
```

the application rejected the person as:

**Unknown Person**

This demonstrates that the system does not simply force every input face into one of the five known player classes.

---

## 🖥️ Streamlit Application

The final application provides:

- Image upload
- Face detection
- Single-face validation
- Detection score
- 512-D embedding generation
- SVM player prediction
- Cosine similarity
- Known/unknown decision
- Supported-player sidebar
- Identification result
- Matching details

### Application Flow

```text
Upload Image
     ↓
Detect Face
     ↓
Check Face Count
     ↓
Generate 512-D Embedding
     ↓
SVM Prediction
     ↓
Calculate Cosine Similarity
     ↓
Compare with 0.50 Threshold
     ↓
Display Result
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application/UI |
| InsightFace | Face detection and face embeddings |
| ArcFace-style embeddings | Facial feature representation |
| Scikit-learn | SVM and cosine similarity |
| NumPy | Numerical processing |
| Pandas | Data processing |
| OpenCV | Image reading and processing |
| Joblib | Loading the trained SVM model |
| ONNX Runtime | InsightFace model inference |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Indian-Cricket-Face-Identification.git
```

Move into the project:

```bash
cd Indian-Cricket-Face-Identification
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## ▶️ How to Use

1. Open the Streamlit application.
2. Upload a JPG, JPEG, or PNG image.
3. The system detects faces.
4. If no face is detected, the image is rejected.
5. If multiple faces are detected, the system asks for a single-person image.
6. For exactly one face, a 512-dimensional embedding is generated.
7. The SVM predicts the player.
8. Cosine similarity is calculated against the player centroids.
9. If the similarity is below `0.50`, the person is rejected as unknown.
10. Otherwise, the identified player is displayed.

---

## 🚀 Deployment

The application can be deployed using **Streamlit Community Cloud**.

Deployment requirements:

```text
app.py
requirements.txt
models/
├── face_embeddings.csv
└── svm_face_classifier.pkl
```

The raw training datasets and processed training images are not required by the live application because the application uses the already trained SVM model and stored embeddings.

### Deployment Workflow

```text
Local Project
      ↓
GitHub Repository
      ↓
Streamlit Community Cloud
      ↓
Live Web Application
```

---

## ⚠️ Limitations

- The current application supports five selected players.
- The system expects an image containing one clearly visible face.
- Performance can vary with poor lighting, blur, occlusion, extreme poses, or low-resolution images.
- The unknown threshold of `0.50` is a project heuristic and has not been established as a universal recognition threshold.
- The reported evaluation results are based on the datasets and test samples used in this project.
- The external validation sample contained only five images.
- The model should not be interpreted as a general-purpose identity verification system.

---

## 🔮 Future Enhancements

Possible future improvements include:

- Add more Indian cricket players.
- Increase the diversity and size of the training dataset.
- Add more independent unknown-person test images.
- Perform formal threshold calibration using separate positive and negative validation sets.
- Add face bounding-box visualization.
- Add prediction history.
- Add a database for storing predictions.
- Add an administrator dashboard.
- Improve handling of multiple faces.
- Deploy using a scalable cloud infrastructure.
- Add automated model retraining when new labeled data becomes available.

---

## 📈 Final Project Results

| Evaluation | Result |
|---|---:|
| Final embedding images | 274 |
| Embedding dimension | 512 |
| Random held-out SVM test | 100% |
| Random held-out test size | 55 |
| Cross-source SVM test | 100% |
| Cross-source test size | 158 |
| Cross-source cosine test | 100% |
| External player test | 5/5 |
| Unknown-person test | Rejected |
| Unknown threshold | 0.50 |

---

## 👥 Project Summary

This project demonstrates an end-to-end computer vision and machine learning workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Face Detection
      ↓
Feature Extraction
      ↓
512-D Embeddings
      ↓
SVM Classification
      ↓
Cosine Similarity
      ↓
Unknown Detection
      ↓
Streamlit Deployment
```

The project combines a pretrained face-recognition model with a traditional machine-learning classifier to create a practical cricket-player identification application.

---

## 📄 License

This repository is intended as an educational/project implementation.

Dataset images remain subject to the licenses and terms of their original sources. The project should not be used to make high-stakes identity decisions.

---

## 👩‍💻 Project

**Indian Cricket Team Member Face Identification**

Built using Python, InsightFace, SVM, cosine similarity, and Streamlit.
