# 📈 Stock Market Unsupervised Clustering Using K-Means

An unsupervised machine learning project that clusters stocks based on their **daily return behavior** using K-Means. Historical price data is fetched dynamically via `yfinance`, making the pipeline fully self-contained — no CSV files required.

---

## 🗂️ Project Structure

```
StockMarket_Clustering/
└── Notebook/
    └── Stock_market_unsupervised_clustering.ipynb
```

---

## 🎯 Objective

Group a curated set of stocks (tech, finance, consumer, travel, and pharma sectors) by how similarly their daily returns behave — without using any labels. The goal is to let the algorithm discover natural market clusters driven purely by price movement patterns.

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Source** | Yahoo Finance (via `yfinance`) |
| **Tickers** | `AAPL`, `META`, `NVDA`, `TSLA`, `ABBV`, `MCD`, `CCL`, `MSFT`, `GS`, `JPM` |
| **Period** | Last **2 years** from the date of execution |
| **Features** | Daily percentage returns computed from closing prices |
| **No static file** | Data is downloaded live inside the notebook |

---

## 🔬 Methodology & Workflow

The notebook is structured into **26 clearly documented steps**:

### 1. Data Acquisition
- Stock tickers are defined and sorted alphabetically.
- A 2-year lookback window is set dynamically using `datetime`.
- Data is downloaded with `yf.download()`, including OHLCV fields for all tickers.

### 2. Exploratory Data Analysis (EDA)
| Step | Analysis |
|---|---|
| Dataset shape & date range | Verifies completeness of the download |
| Missing value check | Identifies any gaps in price data |
| Open & Close price preview | Inspects the raw structure |
| Closing price trend plot | Visual comparison of stock prices over time |

### 3. Feature Engineering
- **Daily % Returns**: Calculated using `pct_change()` on closing prices. Returns are preferred over raw prices because they normalize for absolute price levels.
- **Volatility Analysis**: Standard deviation of returns identifies high-volatility stocks (TSLA stands out clearly).
- **Correlation Matrix**: Heatmap shows how similarly different stocks move day-to-day.
- **Return Summary Stats**: Mean, std, min, and max returns are tabulated per stock.

### 4. Machine Learning Pipeline

```
Raw Close Prices → Daily % Returns → Drop NaN → Feature Matrix (stocks × days)
    → StandardScaler → KMeans (k-means++) → Cluster Labels
```

- **Feature Matrix**: Each stock is represented as a row vector of its daily returns (`shape: [n_stocks, n_days]`).
- **StandardScaler**: Applied before K-Means since it is distance-based; scaling ensures fair comparison across stocks.

### 5. Optimal K Selection
Two complementary metrics are evaluated for `k ∈ [2, 10]`:

| Metric | Role |
|---|---|
| **Inertia** | Measures cluster compactness; visualized via the Elbow Method |
| **Silhouette Score** | Measures cluster separation quality; used to select the best K |

> **Result:** `K = 2` yields the highest silhouette score.

### 6. Final Clustering & Interpretation
- Final K-Means model is trained with `k-means++` initialization, `n_init=12`, and `max_iter=1000`.
- Cluster assignments are joined back to ticker symbols.
- **Cluster 0**: 9 stocks (AAPL, META, NVDA, ABBV, MCD, CCL, MSFT, GS, JPM)
- **Cluster 1**: TSLA — isolated due to its significantly higher volatility and distinct return pattern.

### 7. Visualization
| Plot | Purpose |
|---|---|
| Elbow Method | Inertia vs. K to support cluster selection |
| Silhouette Score Plot | Score vs. K to confirm best K |
| Cluster Size Bar Chart | Distribution of stocks across clusters |
| PCA Scatter Plot (2D) | K-Means clusters projected to 2D using PCA |
| 20-Day Rolling Return Plot | Compares average return behavior of each cluster over time |

> **Note:** PCA is used **only for visualization**, not for clustering.

---

## 🧠 Key Findings

- **K = 2** is the optimal number of clusters based on silhouette score.
- **TSLA is separated** into its own cluster because its return volatility and magnitude are significantly higher than all other selected stocks.
- The clustering result, while imbalanced, is **statistically meaningful** — the algorithm detected a true behavioral outlier rather than producing arbitrary groupings.
- The 20-day rolling return plot confirms that the TSLA cluster shows stronger and more erratic fluctuations compared to the broad market cluster.

---

## 💡 Key Learnings

- Live stock data extraction using `yfinance`
- Daily return-based feature engineering for financial data
- Volatility and correlation analysis as pre-clustering EDA
- K-Means clustering with `k-means++` initialization
- Elbow Method and Silhouette Score for optimal K selection
- PCA as a visualization tool (not a feature reduction step for clustering)
- Interpreting imbalanced but meaningful cluster outcomes

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `yfinance` | Live stock data download |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | StandardScaler, KMeans, PCA, silhouette_score |
| `datetime` | Dynamic date range calculation |

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install yfinance pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the Notebook
Open and run the notebook end-to-end:
```
Notebook/Stock_market_unsupervised_clustering.ipynb
```

> The notebook fetches live data on execution. Results may vary slightly depending on the run date due to the rolling 2-year window.

---

## 📌 Notes

- The ticker list can be freely modified in **Cell 7** of the notebook.
- The lookback window (currently 2 years) can be adjusted in **Cell 9**.
- All steps are self-documented with markdown cells — no external documentation is needed to follow along.
