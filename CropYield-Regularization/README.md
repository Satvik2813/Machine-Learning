# 🌾 Crop Yield Prediction with Regularization

A machine learning project that predicts crop yields across different countries and crop types using regularization techniques (Ridge and Lasso regression). The model leverages agricultural and environmental features like rainfall, pesticide usage, and temperature to estimate crop yields.

---

## 📁 Project Structure

```
CropYield-Regularization/
├── data/
│   └── yield_df.csv          # Dataset with global crop yield records
└── model/
    └── cropyield_Regularization.ipynb  # Main Jupyter Notebook
```

---

## 📊 Dataset Overview

**File:** `data/yield_df.csv`

| Feature | Description |
|---|---|
| `Area` | Country name (101 unique countries) |
| `Item` | Crop type (10 unique crops) |
| `Year` | Year of record (1990–2013) |
| `hg/ha_yield` | Crop yield in hectograms per hectare *(target variable)* |
| `average_rain_fall_mm_per_year` | Annual average rainfall (mm) |
| `pesticides_tonnes` | Pesticide usage in tonnes |
| `avg_temp` | Average annual temperature (°C) |

- **Total records:** 28,242
- **Missing values:** None (clean dataset)

### Crops Covered
Maize, Potatoes, Rice (paddy), Sorghum, Soybeans, Wheat, Sweet potatoes, Cassava, Plantains and others, Yams

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Inspected dataset shape, dtypes, and missing values
- Visualized the target variable (`hg/ha_yield`) distribution — found **right-skewed** distribution (skewness: ~1.79)
- Applied **log1p transformation** on yield to normalize distribution (skewness reduced to ~-0.05)
- Generated a correlation heatmap for numeric features

### 2. Preprocessing
- **Numerical features:** Median imputation via `SimpleImputer`
- **Categorical features:** Mode imputation + One-Hot Encoding (`OneHotEncoder`)
- Combined using `ColumnTransformer` inside a `Pipeline`

### 3. Modeling

| Model | Description |
|---|---|
| **Ridge Regression** | L2 regularization; penalizes large coefficients |
| **RidgeCV** | Cross-validated Ridge to select best alpha |
| **Lasso Regression** | L1 regularization; performs feature selection |
| **LassoCV** | Cross-validated Lasso to select best alpha |

- **Train/Test Split:** 80% training, 20% testing (`random_state=42`)
- Models were wrapped in full `sklearn` pipelines for clean preprocessing + prediction

### 4. Evaluation

Models are evaluated using:
- **R² Score** — measures proportion of variance explained
- **RMSE** — root mean squared error (in log-scale)
- **MAE** — mean absolute error

---

## ⚙️ Technologies Used

| Tool | Version |
|---|---|
| Python | 3.x |
| pandas | Latest |
| numpy | Latest |
| matplotlib | Latest |
| seaborn | Latest |
| scikit-learn | Latest |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Notebook

1. Clone or download this repository.
2. Place `yield_df.csv` inside the `data/` folder (or adjust the path in the notebook).
3. Open `model/cropyield_Regularization.ipynb` using Jupyter Notebook or Google Colab.
4. Run all cells sequentially.

> **Note:** This notebook was originally developed in **Google Colab**. You may need to adjust the file path when reading the CSV locally.

---

## 📈 Key Findings

- The raw yield distribution is **highly right-skewed** (skewness ≈ 1.79), and log transformation brings it close to normal.
- **Pesticide usage** shows the strongest positive correlation with yield among numeric features.
- **Average temperature** has a slight negative correlation with yield.
- Regularization techniques (Ridge, Lasso) help prevent overfitting given the high dimensionality introduced by one-hot encoding of countries and crop types.

---

## 👤 Author

**Satvik**  
Developed with ❤️ on Google Colab.