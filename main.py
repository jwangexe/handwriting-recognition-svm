import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data = pd.read_csv("traindata.csv")

x, y = data.iloc[:, 1:], data.iloc[:, 0]
print(x.head(), y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)