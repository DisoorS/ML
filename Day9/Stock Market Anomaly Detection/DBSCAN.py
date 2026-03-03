import pandas as pd
from sklearn.cluster import DBSCAN

df = pd.read_csv("pca_data.csv")
X=df[["PCA1", "PCA2"]]
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)
cluster = dbscan.labels_
df["Cluster"] = cluster
anomalies = cluster == -1
df["Anomaly"] = anomalies.astype(int)
# import matplotlib.pyplot as plt

# plt.figure(figsize=(8,6))

# normal = df[df["Anomaly"] == 0]
# anomaly = df[df["Anomaly"] == 1]

# plt.scatter(normal["PCA1"], normal["PCA2"])
# plt.scatter(anomaly["PCA1"], anomaly["PCA2"], marker="x", s=100)

# plt.xlabel("PCA1")
# plt.ylabel("PCA2")
# plt.title("Anomaly Detection using GMM")
# plt.show