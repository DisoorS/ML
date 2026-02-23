import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import minmax_scale

df = pd.read_csv("preprocessedDataregression.csv")
df.drop(columns=["Unnamed: 0"] , inplace=True)
X = df.drop("Price", axis = 1)
y = df["Price"]

target_column = "Price"
corr_matrix = df.corr(numeric_only=True)

target_corr = corr_matrix[target_column].abs().sort_values(ascending=False)
print("Correlation with target:\n")
print(target_corr)

k = 4
best_features = target_corr.index[1:k+1]
print("best features:", best_features)

new_df = df[list(best_features) + [target_column]]

new_df.to_csv("selected_feature_regression.csv", index=False)