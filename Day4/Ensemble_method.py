#voting classifier code

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("preprocessedDataclassification.csv")
X = df.drop("sold", axis = 1)
y = df["sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model1 = LogisticRegression(max_iter=1000)
model2 = DecisionTreeClassifier()
model3 = SVC(probability=True)

voting = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('svm', model3)],
    voting='soft'
)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print("voting score:", accuracy_score(y_test, y_pred))

#bagging code

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
print("Bagging accuracy:", accuracy_score(y_test, y_pred))

#randomforest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("randomforest accuracy:", accuracy_score(y_test, y_pred))

#adaboosting (boosting)

from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
print("adaboost accuracy:", accuracy_score(y_test, y_pred))

#Stacking

from sklearn.ensemble import StackingClassifier

base_models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()
)

stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print("stack accuracy:", accuracy_score(y_test, y_pred))