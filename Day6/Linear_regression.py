import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("preprocessedDataregression.csv")

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("mar :",mean_absolute_error(y_test, y_pred))
print("mse :",mean_squared_error(y_test, y_pred))
print("r2 :",r2_score(y_test, y_pred))
