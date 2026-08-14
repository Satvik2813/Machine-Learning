# 🏦 Loan Default Classifier

A machine learning project that predicts whether a loan applicant is likely to default, based on a rich set of financial and demographic features. Built using Python in a Jupyter Notebook environment (Google Colab).

---

## 📁 Project Structure

```
LoanDefaultClassifier/
├── Data/
│   └── Loan_default.csv          # Dataset (≈24 MB)
└── Model/
    └── LoanDefaultPredictor_Classification.ipynb  # Main notebook
```

---

## 📊 Dataset Overview

**File:** `Data/Loan_default.csv`  
**Size:** ~24.8 MB | **Rows:** 255,347 | **Columns:** 18

### Features

| Feature | Type | Description |
|---|---|---|
| `LoanID` | Object | Unique loan identifier |
| `Age` | Integer | Applicant age (18–69) |
| `Income` | Integer | Annual income ($15,000–$149,999) |
| `LoanAmount` | Integer | Loan amount requested ($5,000–$249,999) |
| `CreditScore` | Integer | Credit score (300–849) |
| `MonthsEmployed` | Integer | Duration of current employment (0–119 months) |
| `NumCreditLines` | Integer | Number of active credit lines (1–4) |
| `InterestRate` | Float | Interest rate on loan (2%–25%) |
| `LoanTerm` | Integer | Loan duration in months (12, 24, 36, 48, 60) |
| `DTIRatio` | Float | Debt-to-Income ratio (0.1–0.9) |
| `Education` | Object | Education level (High School, Bachelor's, Master's, PhD) |
| `EmploymentType` | Object | Employment status (Full-time, Part-time, Self-employed, Unemployed) |
| `MaritalStatus` | Object | Marital status (Single, Married, Divorced) |
| `HasMortgage` | Object | Whether applicant has a mortgage (Yes/No) |
| `HasDependents` | Object | Whether applicant has dependents (Yes/No) |
| `LoanPurpose` | Object | Purpose of loan (Auto, Business, Education, Home, Other) |
| `HasCoSigner` | Object | Whether loan has a co-signer (Yes/No) |
| `Default` | Integer | **Target** — 1 if defaulted, 0 otherwise |

### Key Statistics

- **No missing values** across all 255,347 records
- **Default rate:** ~11.6% (imbalanced classification problem)
- **Average income:** ~$82,499
- **Average loan amount:** ~$127,579
- **Average credit score:** ~574

---

## 🔍 Notebook Workflow

### 1. Data Loading & Exploration
- Load CSV dataset using Pandas
- Inspect shape, data types, and null values (`df.head()`, `df.info()`, `df.isnull().sum()`, `df.describe()`)

### 2. Exploratory Data Analysis (EDA)
- Distribution plots for numeric features (Age, Income, CreditScore, etc.)
- Bar charts and count plots for categorical variables
- Target variable distribution analysis (class imbalance check)
- Correlation analysis between features and default status

### 3. Data Preprocessing
- Encoding of categorical features (`Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`)
- Feature scaling / normalization
- Handling class imbalance (if applicable)

### 4. Model Building & Training
- Multiple classification algorithms explored (e.g., Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Train/test split for evaluation

### 5. Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix visualization
- Feature importance analysis

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Matplotlib | Visualization |
| Seaborn | Statistical plots |
| Scikit-learn | ML models & preprocessing |
| Google Colab | Development environment |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run the Notebook

1. Clone or download the repository
2. Place `Loan_default.csv` in the `Data/` directory
3. Open `Model/LoanDefaultPredictor_Classification.ipynb` in Jupyter or Google Colab
4. Run all cells sequentially

> **Note:** If using Google Colab, upload the dataset or mount your Google Drive before running the notebook.

---

## 📈 Business Context

Loan default prediction is a critical task for financial institutions. Accurate models help:
- **Reduce financial risk** by identifying high-risk applicants early
- **Improve lending decisions** using data-driven insights
- **Optimize portfolio management** through risk segmentation

This project demonstrates how standard ML techniques can be applied to real-world financial data to build an effective credit risk classifier.

---

## 📝 Author

**Satvik**

---

## 📄 License

This project is for educational purposes.
