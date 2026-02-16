from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X = data.data
y = data.target

knn = KNeighborsClassifier(n_neighbors=3)
print(knn.fit(X, y))
print(knn.predict(X))