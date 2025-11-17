import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Load Dataset
df = pd.read_csv("data.csv")  # Replace with your dataset path

X = df.iloc[:, :]  # Use all columns for clustering

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ KMeans ------------------
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
print("KMeans Labels:\n", labels_kmeans)

# ------------------ DBSCAN ------------------
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
print("\nDBSCAN Labels:\n", labels_dbscan)

# ------------------ Agglomerative Clustering ------------------
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X_scaled)
print("\nAgglomerative Clustering Labels:\n", labels_agg)

# ------------------ Gaussian Mixture Model (GMM) ------------------
gmm = GaussianMixture(n_components=3, random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)
print("\nGMM Labels:\n", labels_gmm)
