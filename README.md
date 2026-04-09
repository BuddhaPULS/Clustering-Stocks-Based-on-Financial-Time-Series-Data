# Clustering Stocks Based on Financial Time Series Data

A machine learning project that clusters stocks in the **CSI A50 Index** using financial time series data, comparing Dynamic Time Warping (DTW) and Extended Frobenius Norm (EROS) as distance measures for K-means clustering.

---

## 📌 Project Overview

This project explores how to group stocks based on their financial behaviour over time. Rather than comparing raw financial figures (which vary dramatically across companies and quarters), the approach derives **standardized financial ratios** that enable fair cross-company comparison.

**Dataset:** 50 companies in the CSI A50 Index, quarterly financial data from **2020-01-01 to 2024-12-31**, collected via [AKShare](https://github.com/akfamily/akshare).

---

## 🔢 Financial Ratios

Raw financial data is transformed into 8 standardized ratios:

| Abbreviation | Definition |
|---|---|
| `DE` | Debt-to-Equity Ratio (Total Liability / Net Worth) |
| `NCOS` | Net Cash from Operations / Sales |
| `PNNIS` | Net Profit Excl. Non-Recurring Items / Total Operating Revenue |
| `NDN` | (Net Income − Dividends) / Net Income |
| `STA` | Sales / Net Worth (proxy for Total Assets) |
| `ICSNW` | Net Profit Excl. Non-Recurring Items / Net Worth |
| `NITN` | Net Income Attributable to Parent Company / Net Income |
| `CASH_PER_SHARE` | Net Cash Flow per Share |

---

## 📊 Exploratory Data Analysis

Distribution and correlation analysis revealed several key findings:

- **DE** is concentrated between 0–2 for most companies, with a few extreme outliers near 10.
- **NCOS** centers around 0 with notable extreme values.
- **NDN** shows anomalous values greater than 1, indicating potential data quality issues.
- Three notable feature correlations were found:
  - `PNNIS` & `ICSNW`: *r* = 0.50 (shared numerator)
  - `STA` & `ICSNW`: *r* = 0.37 (shared denominator)
  - `PNNIS` & `STA`: *r* = −0.30 (structural inverse relationship)

---

## ⚙️ Methods

Since the data is **multivariate time series**, standard Euclidean distance is insufficient. The pipeline uses the following algorithms:

### 1. Dynamic Time Warping (DTW)
Measures similarity between two time series by finding the optimal temporal alignment through "stretching" or "compressing" the time axis. Implemented via dynamic programming:

$$DTW[i,j] = d(s[i], t[j]) + \min(DTW[i-1,j],\ DTW[i,j-1],\ DTW[i-1,j-1])$$

**Limitations addressed:**
- Aggregation strategy applied to handle multivariate series.
- Recency weighting considered, as recent financial data carries more relevance.

### 2. Extended Frobenius Norm (EROS)
A PCA-based similarity measure for multivariate time series. For two MTS **X** and **Y**:

1. Perform SVD: $X = U_X \Sigma_X V_X^T$, $Y = U_Y \Sigma_Y V_Y^T$
2. Compute cosine similarity between corresponding eigenvectors: $s_i = \langle u_i, v_i \rangle$
3. Weight by normalized eigenvalues:

$$EROS(X, Y) = \sum_{i=1}^{n} w_i s_i, \quad w_i = \frac{\lambda_{Yi} + \lambda_{Xi}}{\sum_{j=1}^{n} \lambda_{Yj} + \lambda_{Xj}}$$

### 3. Multidimensional Scaling (MDS)
Converts the pairwise distance matrix into Euclidean coordinates suitable for K-means. Uses double-centering to form a Gram matrix **B**, then reconstructs coordinates via eigen-decomposition. Embedding quality is measured by **strain**:

$$\mathrm{Strain}(X) = \left(\frac{\sum_{i,j}(b_{ij} - x_i^\top x_j)^2}{\sum_{i,j} b_{ij}^2}\right)^{1/2}$$

### 4. K-means Clustering
Applied on the MDS-embedded coordinates. Optimal cluster count *k* is selected by **silhouette score** (evaluated over *k* = 2 to number of features).

---

## 📈 Results

### DTW + K-means
- Optimal *k* = **6** with silhouette score = **0.39**
- Clustering quality was poor — 2 clusters contained only 1–2 companies each, with the remaining clusters not well-separated.

### EROS + K-means
- Optimal *k* = **4** with silhouette score = **0.60**
- All 4 clusters were clearly separated, indicating much better grouping quality.

### Conclusion
**EROS-based K-means outperforms DTW-based K-means** for clustering stocks using financial time series data. The higher silhouette score and cleaner cluster separation make EROS the recommended distance measure for this task.

---

## 🗂️ Project Structure

```
├── notebooks/          # Exploratory analysis and modelling notebooks
└── README.md
```

---

## 🛠️ Dependencies

- [AKShare](https://github.com/akfamily/akshare) — financial data collection
- `numpy`, `pandas` — data processing
- `scikit-learn` — K-means, MDS, silhouette scoring
- `matplotlib`, `seaborn` — visualisation
- `dtaidistance` or equivalent — DTW computation

---

## 👤 Author

**Lin Hengzhou** — November 2025
