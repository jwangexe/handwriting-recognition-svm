import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Collecting data")
data = pd.read_csv("traindata.csv")

x, y = data.iloc[:, 1:], data.iloc[:, 0]
print(x.head(), y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

print("Fitting model")
svm = SVC()
svm.fit(x_train, y_train)

print("Making predictions")
y_pred = svm.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)
print(f"Accuracy: {accuracy}")