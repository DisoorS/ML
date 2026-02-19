import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

df = pd.read_csv("preprocessedDataregression.csv")

X = df.select_dtypes(include=["int64", "float64"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

db_scan = DBSCAN(eps=0.5, min_samples=4)

clusters = db_scan.fit_predict(X_scaled)

df["Cluster"] = clusters
print(df.head())