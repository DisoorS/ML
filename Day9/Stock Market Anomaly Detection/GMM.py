import pandas as pd
from sklearn.covariance import log_likelihood
from sklearn.mixture import GaussianMixture

df = pd.read_csv("pca_data.csv")
X=df[["PCA1", "PCA2"]]
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

cluster = gmm.predict(X)
df["Cluster"] = cluster
log_livelihood = gmm.score_samples(X)
log_likelihood_mean = log_livelihood.mean()
log_likelihood_std = log_livelihood.std()
print(log_likelihood_mean, log_likelihood_std)
threshold = log_likelihood_mean - 2 * log_likelihood_std
anomalies = log_livelihood < threshold

df["Anomaly"] = anomalies.astype(int)

total_anomalies = df["Anomaly"].sum()
print(f"Total anomalies detected: {total_anomalies}")
total_clusters = df["Cluster"].nunique()
print(f"Total clusters formed: {total_clusters}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

normal = df[df["Anomaly"] == 0]
anomaly = df[df["Anomaly"] == 1]

plt.scatter(normal["PCA1"], normal["PCA2"])
plt.scatter(anomaly["PCA1"], anomaly["PCA2"], marker="x", s=100)

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("Anomaly Detection using GMM")
plt.show()