import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("preprocessedDataclassification.csv")

X = df.drop("sold", axis=1)
y = df["sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.2, stratify=y, random_state=42
)

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("accuracy :", accuracy_score(y_test, y_pred))
print("classification report :", classification_report(y_test, y_pred))