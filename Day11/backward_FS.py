import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

df = pd.read_csv("C:\\Users\\DELL\\Music\\ML\\preprocessedDataclassification.csv")
X = df.drop('sold', axis=1)
y = df['sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

sfs_backward = SFS(model,
                   k_features='best',
                   forward=False,
                   floating=False,
                   scoring='accuracy',
                   cv=5)

sfs_backward.fit(X_train, y_train)

selected_features_backward = list(sfs_backward.k_feature_names_)
print("Selected Features (Backward):", selected_features_backward)