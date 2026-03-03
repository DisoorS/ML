import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

df = pd.read_csv("C:\\Users\\DELL\\Music\\ML\\preprocessedDataclassification.csv")
X = df.drop('sold', axis=1)
y = df['sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

sfs = SFS(model,
          k_features='best',
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=5)

sfs.fit(X_train, y_train)

selected_features = list(sfs.k_feature_names_)
print("Selected Features:", selected_features)