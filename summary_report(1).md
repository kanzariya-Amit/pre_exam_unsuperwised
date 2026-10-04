# Customer Segmentation using Unsupervised Learning

This project performs customer segmentation using the **Online Retail II** dataset. The analysis focuses on customers from the **United Kingdom**. The main objective is to identify groups of customers with similar purchasing behaviour and compare different unsupervised learning algorithms.

## 1. Data Preparation and Exploration

First, the dataset was loaded and explored by checking its shape, columns, data types and sample records. The data was filtered to include only UK customers. Rows with missing `CustomerID`, `Quantity` less than or equal to zero, and `UnitPrice` less than or equal to zero were removed.

A new `TotalPrice` feature was created:

`TotalPrice = Quantity × UnitPrice`

Exploratory analysis was performed using histograms, monthly transaction analysis, customer spending, frequent buyers and Pareto analysis.

## 2. RFM Feature Engineering

For customer segmentation, **RFM analysis** was performed:

- **Recency:** Number of days since the customer's most recent purchase.
- **Frequency:** Number of unique invoices for the customer.
- **Monetary:** Total amount spent by the customer.

Extreme values were handled using the **Q3 + 3×IQR** capping method. Frequency and Monetary were transformed using `log1p` to reduce skewness.

The final clustering features were `Recency`, `Frequency_log`, and `Monetary_log`, followed by `StandardScaler`.

## 3. K-Means Clustering

K-Means was tested for `k = 2` to `10` using the elbow method and silhouette score. A final value of **k = 4** was selected because it provides useful business-level segmentation with four interpretable groups.

The final model used `k-means++`, `n_init=20`, `max_iter=500`, and `random_state=42`.

The four customer groups were interpreted as:

1. **Champions** – Recent, frequent and high-value customers.
2. **Loyal Customers** – Regular customers with meaningful spending.
3. **At-Risk / Hibernating Customers** – Customers with higher recency and lower purchasing activity.
4. **Low-Engagement Customers** – Customers with low purchase frequency and comparatively low monetary value.

## 4. Agglomerative Clustering

Agglomerative Clustering was evaluated using **Ward, Complete and Average linkage** for 3, 4 and 5 clusters.

The selected solution used **4 clusters with Ward linkage**. A dendrogram was created to understand the hierarchical structure and support the cluster selection.

## 5. DBSCAN Clustering

DBSCAN was evaluated using a **k-nearest-neighbour distance curve** and a parameter grid:

- `eps = [0.3, 0.5, 0.7, 1.0, 1.5]`
- `min_samples = [3, 5, 8, 10]`

Silhouette scores were calculated using non-noise observations. DBSCAN also identified some customers as noise, representing unusual RFM patterns that did not belong to dense customer groups.

## 6. Model Comparison

The algorithms were compared using:

- **Silhouette Score** – higher is better.
- **Davies-Bouldin Index** – lower is better.
- **Calinski-Harabasz Score** – higher is better.
- **Noise points** – relevant for DBSCAN.

K-Means provided a strong combination of clustering quality, stability and business interpretability. K-Means stability was also checked using seeds `0, 7, 21, 42, 99`.

## 7. Business Recommendation

| Segment | Recommended Action |
|---|---|
| Champions | VIP rewards, loyalty benefits, early access and referral campaigns |
| Loyal Customers | Cross-selling, upselling and personalized recommendations |
| At-Risk / Hibernating | Win-back offers, reminders and reactivation campaigns |
| Low-Engagement Customers | Targeted activation offers and repeat-purchase campaigns |

Additional data such as product preferences, demographics, marketing channels, discount usage, returns, website/app engagement and geography could improve future segmentation.

## 8. Model Deployment

The trained `StandardScaler` and K-Means model were saved using **joblib**:

- `rfm_scaler.pkl`
- `kmeans_customer_segmentation.pkl`

A `predict_segment()` function was created to accept raw **Recency, Frequency and Monetary** values and return the predicted customer cluster and persona.

Five hypothetical customers were tested to demonstrate how new customers can be assigned to an existing segment.

## Conclusion

This project demonstrates how unsupervised learning can convert transaction-level retail data into meaningful customer segments. RFM analysis combined with clustering provides a practical way to understand customer purchasing behaviour and create targeted marketing strategies. Among the tested approaches, **K-Means with four clusters** was selected as the operational model because it offered a useful balance of clustering performance, stability and business interpretability.
