# 🌍 Earthquake Zone Clustering using HDBSCAN

An unsupervised machine learning project that applies **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)** to discover natural earthquake zones from global seismic event data.

---

## 📋 Project Overview

This project uses density-based clustering to group earthquake events into meaningful geospatial zones based on their physical characteristics. Since there are no pre-defined labels, the model is evaluated using internal clustering quality metrics.

The notebook is structured in the **Andrew Ng ML Specialization** style — clean, methodical, and well-reasoned at each step.

---

## 📂 Repository Structure

```
EarthQuake clustering/
├── data/
│   └── earthquake_dataset.csv       # Raw earthquake data
└── notebook/
    └── earthquake_zone_clustering_hdbscan.ipynb  # Main analysis notebook
```

---

## 📊 Dataset

**File:** `earthquake_dataset.csv`

| Column | Description |
|---|---|
| `Date` | Date of the earthquake (YYYY-MM-DD) |
| `Time (UTC)` | Time of occurrence in UTC |
| `City` | Nearest city/location description |
| `Country` | Country of occurrence |
| `Latitude` | Geographic latitude of epicenter |
| `Longitude` | Geographic longitude of epicenter |
| `Earthquake Magnitude` | Magnitude on the Richter scale |
| `Depth (km)` | Depth of the earthquake hypocenter |
| `Impact Score` | A composite severity score |

**Dataset Size:** 16,944 records | 9 columns  
**Date Range:** Historical global seismic events (magnitude ≥ 5.0)

---

## 🔬 Workflow

The notebook follows a 10-step structured pipeline:

1. **Install Required Libraries** — Ensures all dependencies are available
2. **Import Libraries** — Loads NumPy, Pandas, scikit-learn, HDBSCAN, Matplotlib, Seaborn
3. **Load Dataset** — Reads the CSV and performs initial inspection
4. **Basic Data Check** — Verifies shape, data types, and confirms **zero missing values**
5. **Select Features** — Chooses numeric features relevant to clustering:
   - `Latitude`, `Longitude` (geospatial location)
   - `Earthquake Magnitude` (strength)
   - `Depth (km)` (hypocenter depth)
   - `Impact Score` (composite severity)
6. **Correlation & Multicollinearity Check** — Generates a heatmap to assess feature relationships
7. **Feature Distribution Visualizations** — Plots histograms and box plots for all selected features
8. **Scale Features** — Applies `StandardScaler` for normalization
9. **Train HDBSCAN** — Fits the model with reasoned parameter choices
10. **Evaluate & Visualize** — Computes internal metrics and creates geographic + PCA cluster plots

---

## 🧰 Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
```

**Install with:**
```bash
pip install pandas numpy scikit-learn hdbscan matplotlib seaborn
```

---

## 📈 Evaluation Metrics

Since this is **unsupervised learning** (no ground truth labels), cluster quality is measured using:

| Metric | Description |
|---|---|
| **Silhouette Score** | Measures cohesion vs. separation (higher is better) |
| **Calinski-Harabasz Score** | Ratio of between-cluster to within-cluster dispersion (higher is better) |
| **Davies-Bouldin Score** | Average similarity between each cluster and its most similar one (lower is better) |

---

## 🔑 Key Findings & Notes

- **Impact Score** has a strong correlation (0.72) with **Earthquake Magnitude**, indicating potential multicollinearity.
- **Latitude/Longitude** use standard Euclidean scaling — a production model would benefit from Haversine distance.
- HDBSCAN is preferred over K-Means or DBSCAN because:
  - It handles **variable-density clusters** common in geospatial data
  - It naturally identifies **noise/outlier events**
  - No need to pre-specify the number of clusters

---

## 🚀 Getting Started

1. **Clone or download** this repository
2. Place `earthquake_dataset.csv` in the `data/` folder
3. Open `notebook/earthquake_zone_clustering_hdbscan.ipynb` in Jupyter or Google Colab
4. Run all cells in order

---

## 👤 Author

**Satvik**  
*Developed as part of an ML learning portfolio*
