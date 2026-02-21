import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz

df = pd.read_csv("preprocessedDataregression.csv")

X = df.select_dtypes(include=["int64", "float64"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

data = X_scaled.T

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data,
    c=3,
    m=2,
    error=0.005,
    maxiter=1000,
    init=None
)
cluster_labels = np.argmax(u, axis=0)

df["Cluster"] = cluster_labels