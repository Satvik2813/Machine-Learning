# 🎬 Content-Based Movie Recommendation System

A content-based filtering recommender system built using **TF-IDF vectorization** and **K-Nearest Neighbors (KNN)** to surface similar movies based on their textual and metadata features.

---

## 📋 Project Overview

This notebook constructs a movie recommendation engine in the **Andrew Ng ML Specialization** style — each movie is converted into an item-feature vector `x(i)`, and similar movies are retrieved using vector similarity. The implementation uses only vectorized pandas and scikit-learn operations (no explicit `for` loops).

---

## 📂 Repository Structure

```
Movie Recommendation System/
├── data/
│   └── Movies.csv                            # Raw movie metadata (~1.4M rows)
└── notebook/
    └── content_based_movie_recommender.ipynb # Main analysis notebook
```

---

## 📊 Dataset

**File:** `Movies.csv`

| Column | Description |
|---|---|
| `id` | Unique movie identifier |
| `title` | Movie title |
| `vote_average` | Average user rating |
| `vote_count` | Number of votes |
| `release_date` | Release date |
| `runtime` | Movie runtime (minutes) |
| `original_language` | Primary language |
| `overview` | Plot summary |
| `popularity` | Popularity score |
| `tagline` | Marketing tagline |
| `genres` | Comma-separated genre list |
| `production_companies` | Production company names |
| `keywords` | Descriptive keywords |

**Dataset Size:** ~1.4 million rows (filtered to `MAX_MOVIES = 50,000` for performance)

---

## 🔬 Workflow

The notebook follows a structured pipeline:

### 1. Setup
Imports all required libraries and defines global configuration:
```python
MAX_MOVIES = 50_000   # Limit for memory efficiency
RANDOM_STATE = 42
```

### 2. Load Data
Reads the large CSV file and selects only the relevant columns.

### 3. Clean Data
- Converts `release_date` to datetime and extracts `release_year`
- Converts numeric fields (`vote_average`, `vote_count`, `popularity`, `runtime`)
- Fills missing text fields with empty strings
- Drops duplicate and untitled entries
- Normalizes `title_clean` for case-insensitive lookups

### 4. Exploratory Data Analysis (EDA)
Visual analysis of the dataset:
- **Movie Count by Release Year** — Distribution of releases over time
- **Top 15 Genres** — Most frequent genre categories in the corpus
- **Vote Average Distribution** — Rating quality across the dataset

### 5. Feature Engineering
Combines textual fields into a single **content string** per movie:
```
content = genres + keywords + overview + tagline + language + production_companies
```

### 6. TF-IDF Vectorization
Transforms the content string into a high-dimensional feature matrix using `TfidfVectorizer`. L2 normalization is applied to all vectors.

### 7. KNN Model Training
Fits a **Nearest Neighbors** model (cosine similarity) on the TF-IDF feature matrix.

### 8. Recommendation Function
Given any movie title, the system:
1. Looks up the movie's feature vector
2. Queries the KNN model for the `k` most similar movies
3. Returns the top recommendations with metadata

---

## 🧰 Libraries Used

```python
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
```

**Install with:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🎯 Recommendation Logic

Each movie is represented by a **content vector** built from:

| Feature | Weight / Rationale |
|---|---|
| `genres` | High signal for category similarity |
| `keywords` | Captures thematic elements |
| `overview` | Rich natural language description |
| `tagline` | Captures tone and style |
| `original_language` | Groups same-language productions |
| `production_companies` | Studio/franchise similarity |

Similarity is measured via **cosine distance** between normalized TF-IDF vectors.

---

## 📈 Example Usage

```python
# Get top 10 recommendations for "Inception"
recommend("Inception", n=10)
```

**Sample output:**
```
1. Interstellar        (2014) — Adventure, Drama, Science Fiction
2. The Dark Knight     (2008) — Drama, Action, Crime, Thriller
3. Memento             (2000) — Mystery, Thriller
...
```

---

## ⚙️ Performance Considerations

- The dataset contains **1.4M+ rows** — it is subsampled to `MAX_MOVIES = 50,000` by default.
- Increase `MAX_MOVIES` if your machine has sufficient RAM.
- The TF-IDF matrix can be memory-intensive at large scale; consider sparse matrix operations for production use.

---

## 🔑 Key Design Decisions

- **Content-Based (not Collaborative):** No user ratings or watch history required.
- **No explicit loops:** All operations use vectorized pandas / scikit-learn APIs.
- **Cosine similarity:** Preferred over Euclidean distance for text data.
- **Subsampling:** The notebook limits to 50K movies for interactive use — adjustable via `MAX_MOVIES`.

---

## 🚀 Getting Started

1. **Clone or download** this repository
2. Place `Movies.csv` in the `data/` folder
3. Open `notebook/content_based_movie_recommender.ipynb` in Jupyter or Google Colab
4. Run all cells in order

> **Note:** The CSV file is large (~630MB). Ensure sufficient disk space and RAM before loading.

---

## 👤 Author

**Satvik**  
*Developed as part of an ML learning portfolio*
