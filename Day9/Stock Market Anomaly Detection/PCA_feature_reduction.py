import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
df = pd.read_csv("Day9\Stock Market Anomaly Detection\preprocessed_data.csv")
X = df
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=X_pca, columns=["PCA1", "PCA2"])
pca_df.to_csv("pca_data.csv", index=False)
print(pca.explained_variance_ratio_)
plt.figure(figsize=(8,6))
plt.scatter(pca_df["PCA1"], pca_df["PCA2"])
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA Feature Reduction")
plt.show()