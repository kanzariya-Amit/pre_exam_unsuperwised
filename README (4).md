# Customer Segmentation using Unsupervised Learning

**Practical Exam – Set A | Online Retail II dataset**

UK customer segmentation using **RFM analysis** (Recency, Frequency, Monetary) and a comparison of three clustering algorithms: **K-Means**, **Agglomerative (Hierarchical) Clustering** and **DBSCAN**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [How to Run](#how-to-run)
6. [Methodology](#methodology)
7. [Customer Personas & Business Actions](#customer-personas--business-actions)
8. [Saved Model & Prediction](#saved-model--prediction)
9. [Future Improvements](#future-improvements)

---

## Project Overview

Customer segmentation groups customers with similar buying behaviour so that marketing can be targeted instead of generic. This project:

- Cleans the Online Retail II transaction data (UK customers only)
- Performs exploratory data analysis (EDA)
- Engineers RFM features per customer
- Applies three clustering algorithms and compares them with multiple metrics
- Recommends a final model (K-Means, 4 segments) with business actions
- Saves the trained model and scaler to predict the segment of new customers

## Dataset

- **Name:** Online Retail II (UCI / Kaggle)
- **File expected:** `online_retail_II.csv` (or `online_retail_II(1).csv`) in the same folder as the notebook
- **Key columns used:** `Invoice`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`
- Columns are renamed to `InvoiceNo`, `UnitPrice`, `CustomerID` inside the notebook.

> The dataset is not included in this repository. Download it and place it next to the notebook.

## Project Structure

```
.
├── Customer_Segmentation_Step1_to_Step7.ipynb   # Main notebook
├── online_retail_II.csv                         # Dataset (add manually)
├── rfm_scaler.pkl                               # Generated: fitted StandardScaler
├── kmeans_customer_segmentation.pkl             # Generated: trained K-Means model
└── README.md
```

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- joblib
- jupyter

Install everything with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib jupyter
```

## How to Run

1. Clone or download this repository.
2. Place `online_retail_II.csv` in the project folder.
3. Start Jupyter:
   ```bash
   jupyter notebook Customer_Segmentation_Step1_to_Step7.ipynb
   ```
4. Run all cells from top to bottom (**Kernel → Restart & Run All**).

## Methodology

### Step 1 – Data Loading, Cleaning & EDA
- Filter to **United Kingdom** transactions
- Remove rows with missing `CustomerID`
- Remove rows with `Quantity <= 0` (returns/cancellations) and `UnitPrice <= 0`
- Create `TotalPrice = Quantity × UnitPrice`
- EDA: distributions (log scale), top 10 countries by orders, monthly UK transactions, customer spend distribution, top customers by frequency, and the share of customers driving 80% of revenue

### Step 2 – RFM Feature Engineering
| Feature | Definition |
|---|---|
| **Recency** | Days between the snapshot date (`2011-12-31`) and the customer's last purchase |
| **Frequency** | Number of unique invoices |
| **Monetary** | Total spend |

Preprocessing:
- Outliers capped using the IQR method (`Q3 + 3 × IQR`)
- `log1p` transform on Frequency and Monetary to reduce skew
- `StandardScaler` applied to `Recency`, `Frequency_log`, `Monetary_log`

### Step 3 – K-Means
- Tested `k = 2 … 10` using **inertia (elbow)** and **silhouette score**
- **Selected k = 4**: although k = 2 gives the highest silhouette, k = 4 provides a more useful business-level segmentation and a reasonable elbow trade-off
- Clusters visualised in 2D (R vs M, F vs M) and 3D RFM space and profiled by mean RFM values

### Step 4 – Agglomerative Hierarchical Clustering
- Ward linkage dendrogram with a 4-cluster cut
- Compared `n_clusters ∈ {3, 4, 5}` and linkage ∈ {ward, complete, average} using silhouette
- Final model: 4 clusters, Ward linkage

### Step 5 – DBSCAN
- k-distance (5-NN) curve for guidance on `eps`
- Grid search over `eps ∈ {0.3, 0.5, 0.7, 1.0, 1.5}` and `min_samples ∈ {3, 5, 8, 10}`
- Best parameters chosen by silhouette (noise excluded); noise points are reported and visualised separately

### Step 6 – Algorithm Comparison
| Metric | Better when |
|---|---|
| Silhouette Score | Higher |
| Davies-Bouldin Index | Lower |
| Calinski-Harabasz Index | Higher |
| Noise points (DBSCAN) | Fewer |

K-Means stability is also checked across 5 random seeds (`0, 7, 21, 42, 99`) and reported as mean ± std of silhouette.

### Step 7 – Model Saving & Prediction
Saves the scaler and K-Means model with `joblib` and provides a `predict_segment(recency, frequency, monetary)` function that is tested on five sample customers.

## Customer Personas & Business Actions

**Recommended operational model: K-Means with 4 interpretable segments.**

| Segment | Description | Suggested Action |
|---|---|---|
| **Champions** | Recent, frequent, high spend | VIP rewards, early access, referral programs |
| **Loyal Customers** | Regular, valuable buyers | Cross-sell and upsell |
| **At-Risk / Hibernating** | Inactive, low frequency | Win-back offers and reminders |
| **Low-Engagement** | Low frequency and low value | Activation campaigns |

> Note: cluster IDs from K-Means can change if the data or random state changes. The notebook's `persona_map` (`{0: Champions, 3: Loyal, 2: At-Risk, 1: Low-Engagement}`) should be re-checked against the cluster profile table after re-running.

## Saved Model & Prediction

After running Step 7, load the artifacts and predict a segment:

```python
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load("rfm_scaler.pkl")
kmeans = joblib.load("kmeans_customer_segmentation.pkl")

def predict_segment(recency, frequency, monetary):
    x = pd.DataFrame({
        "Recency": [recency],
        "Frequency_log": [np.log1p(frequency)],
        "Monetary_log": [np.log1p(monetary)],
    })
    return int(kmeans.predict(scaler.transform(x))[0])

print(predict_segment(recency=25, frequency=20, monetary=6000))
```

Map the returned cluster ID to a persona using the profile table from Step 3.

## Future Improvements

Additional data that could make segmentation more targeted:

- Product category preferences
- Customer demographics
- Acquisition channel
- Discount usage and returns behaviour
- App / website engagement
- Geographic data

---

**Tools:** Python · pandas · scikit-learn · scipy · matplotlib · seaborn
