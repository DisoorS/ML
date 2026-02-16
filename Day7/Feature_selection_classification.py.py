import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import minmax_scale

df = pd.read_csv("preprocessedDataregression.csv")
df.drop(columns=["Unnamed: 0"] , inplace=True)

X = df.drop("Price", axis = 1)
y = df["Price"]

scaler = minmax_scale(X, feature_range=(0, 1))
X_scaled = pd.DataFrame(scaler, columns=X.columns)
X_selected = X_scaled[["Memory", "Company"]]
print(X_selected)
chi2_scores, p_values = chi2(X_selected, y)
print("Chi-squared scores:", chi2_scores)
print("P-values:", p_values)
new_df = pd.concat([X_selected, y], axis=1)
new_df.to_csv("selected_feature_classification.csv", index=False)
