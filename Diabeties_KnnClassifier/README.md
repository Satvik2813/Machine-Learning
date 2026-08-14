# 🩺 Diabetes Prediction - KNN Classifier

A machine learning project that predicts diabetes outcomes using the **K-Nearest Neighbors (KNN)** algorithm. The notebook demonstrates a complete end-to-end ML workflow — from data loading and exploratory analysis to model building, hyperparameter tuning, and evaluation.

---

## 📁 Project Structure

```
Diabeties_KnnClassifier/
├── data/
│   └── diabetes.csv          # Pima Indians Diabetes Dataset
└── notebook/
    └── diabetes_knn.ipynb    # Main Jupyter Notebook
```

---

## 📊 Dataset

**File:** `data/diabetes.csv`

The dataset is the well-known **Pima Indians Diabetes Dataset**, containing diagnostic measurements from **768 female patients** of Pima Indian heritage.

### Features

| Column | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body Mass Index (weight in kg / height in m²) |
| `DiabetesPedigreeFunction` | Genetic diabetes risk score based on family history |
| `Age` | Age in years |
| `Outcome` | Target variable — `1` = Diabetic, `0` = Non-diabetic |

### Class Distribution

| Outcome | Proportion |
|---|---|
| 0 (No Diabetes) | ~65.1% |
| 1 (Diabetes) | ~34.9% |

> ⚠️ Note: The dataset is moderately imbalanced, with roughly 2:1 ratio of non-diabetic to diabetic patients.

---

## 🔬 ML Workflow

The notebook follows a structured 7-step machine learning pipeline:

### 1. 📦 Import Libraries
Essential libraries used:
- `numpy`, `pandas` — Data manipulation
- `matplotlib`, `seaborn` — Data visualization
- `scikit-learn` — ML pipeline, preprocessing, modeling, and evaluation

### 2. 📂 Load Dataset
The dataset is loaded using `pandas` and inspected for structure, shape, and initial patterns.

### 3. 🧹 Treat Zero Values as Missing
Biological features such as `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` cannot realistically be zero. These zero values are treated as missing data (`NaN`) and later imputed using median imputation inside the pipeline.

### 4. 📈 Exploratory Data Analysis (EDA)
Visual analysis includes:
- **Class distribution bar chart** — Illustrates the imbalance between diabetic and non-diabetic patients
- **Feature histograms** — Distribution of each feature split by outcome to identify patterns
- **Correlation heatmap** — Visualizes pairwise feature correlations to detect multicollinearity

### 5. ✂️ Train-Test Split
The dataset is split into training and testing sets using an 80/20 split (`train_test_split`).

### 6. 🤖 Build Preprocessing + KNN Pipeline
A `scikit-learn` `Pipeline` is constructed with:
- **`SimpleImputer`** — Fills missing values (zeros treated as NaN) using the **median** strategy
- **`StandardScaler`** — Normalizes features to zero mean and unit variance (critical for distance-based models like KNN)
- **`KNeighborsClassifier`** — The core classification model

### 7. ⚙️ Hyperparameter Tuning with Cross-Validation
`GridSearchCV` is used to find the optimal value of **k** (number of neighbors) via **5-fold cross-validation**, ensuring robust model selection without overfitting to the test set.

### 8. 📉 Evaluate Model
Final model evaluation includes:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score per class)
- **Confusion Matrix**
- **ROC-AUC Score** and **ROC Curve** — For probabilistic threshold analysis

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed with the following packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the Notebook

1. Clone or download this repository.
2. Navigate to the project directory.
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook notebook/diabetes_knn.ipynb
   ```
4. Run all cells sequentially.

> 📌 Make sure the dataset path in the notebook (`diabetes.csv`) is correct relative to where the notebook is executed.

---

## 🛠️ Key Techniques Used

| Technique | Purpose |
|---|---|
| Zero → NaN conversion | Handling biologically impossible zero values |
| Median Imputation | Robust missing value filling |
| Standard Scaling | Feature normalization for KNN distance calculations |
| Scikit-learn Pipeline | Encapsulates preprocessing + model for clean, leak-free workflow |
| GridSearchCV | Automated hyperparameter tuning with cross-validation |
| ROC-AUC | Threshold-independent model performance metric |

---

## 📌 Notes

- The KNN algorithm is sensitive to feature scaling — `StandardScaler` is essential here.
- Imputation is done **inside** the pipeline to prevent data leakage from the test set into the training process.
- The best `k` is selected via cross-validation to prevent overfitting.

---

## 📄 License

This project is for educational and demonstration purposes.
