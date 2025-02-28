# K-Means Clustering

## Overview
This project implements **K-Means Clustering** using Python. It aims to segment customers based on different features, such as spending habits and annual income. The model clusters similar customers together, providing insights for business applications like targeted marketing.

## Dataset
The dataset used in this project is **Mall_Customers.csv**, which contains:
- **Customer ID** (Unique identifier)
- **Gender**
- **Age**
- **Annual Income (k$)**
- **Spending Score (1-100)**

## Prerequisites
Ensure you have the following libraries installed before running the code:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Installation & Usage
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd K_Means_Clustering
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook K_Means_Clustering.ipynb
   ```

## Steps Involved
1. **Import Libraries**
   - `pandas` for data handling
   - `numpy` for numerical operations
   - `matplotlib` for visualization
   - `scikit-learn` for clustering algorithms

2. **Load Dataset**
   ```python
   df = pd.read_csv('Mall_Customers.csv')
   df.head()
   ```

3. **Finding the Optimal Number of Clusters (Elbow Method)**
   ```python
   from sklearn.cluster import KMeans
   wcss = []
   for i in range(1, 11):
       kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
       kmeans.fit(df.iloc[:, [3, 4]])
       wcss.append(kmeans.inertia_)
   plt.plot(range(1, 11), wcss)
   plt.title('Elbow Method')
   plt.xlabel('Number of Clusters')
   plt.ylabel('WCSS')
   plt.show()
   ```

4. **Train the K-Means Model**
   ```python
   kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
   clusters = kmeans.fit_predict(df.iloc[:, [3, 4]])
   ```

5. **Visualizing the Clusters**
   ```python
   plt.scatter(df.iloc[clusters == 0, 3], df.iloc[clusters == 0, 4], color='red', label='Cluster 1')
   plt.scatter(df.iloc[clusters == 1, 3], df.iloc[clusters == 1, 4], color='blue', label='Cluster 2')
   plt.scatter(df.iloc[clusters == 2, 3], df.iloc[clusters == 2, 4], color='green', label='Cluster 3')
   plt.scatter(df.iloc[clusters == 3, 3], df.iloc[clusters == 3, 4], color='cyan', label='Cluster 4')
   plt.scatter(df.iloc[clusters == 4, 3], df.iloc[clusters == 4, 4], color='magenta', label='Cluster 5')
   plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
   plt.title('Clusters of Customers')
   plt.xlabel('Annual Income (k$)')
   plt.ylabel('Spending Score (1-100)')
   plt.legend()
   plt.show()
   ```

## Results
- The **Elbow Method** helps determine the optimal number of clusters.
- The **K-Means algorithm** segments customers into distinct groups based on their spending habits and annual income.
- The clusters can be used for **customer segmentation** and **business insights**.

## References
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Pandas Documentation](https://pandas.pydata.org/)

## License
This project is licensed under the MIT License.
