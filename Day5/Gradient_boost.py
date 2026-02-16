import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("preprocessedDataclassification.csv")

X = df.drop("sold", axis=1)
y = df["sold"]

x_train , X_test, y_train, y_test = train_test_split(
    X, y , test_size=0.2, stratify=y, random_state=42
)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=1.0,
    max_depth=3,
    random_state=42
)

model.fit(x_train, y_train)
y_pred  = model.predict(X_test)

print("accuracy :", accuracy_score(y_test, y_pred))
print("classification report :", classification_report(y_test, y_pred))