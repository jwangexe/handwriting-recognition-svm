import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv("traindata.csv")

x, y = data.iloc[:, 1:], data.iloc[:, 0]
print(x, y)