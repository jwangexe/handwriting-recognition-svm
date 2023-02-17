import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# implementation of HOG
def calc_hog(x, dimensions=(28, 28), cell_size=(4, 4)):
    fd_list = []
    for row in x:
        img = row.reshape(dimensions)
        fd = hog(img, orientations=8, pixels_per_cell=cell_size, cells_per_block=(1, 1))
        fd_list.append(fd)
    
    return np.array(fd_list)

# reading data from csv file
print("Collecting data")
data = pd.read_csv("traindata.csv")

# separating labels from pixel data
x, y = data.iloc[:, 1:], data.iloc[:, 0]
#print(x.head(), y.head())

# processing x using HOG
print("HOG")
x = calc_hog(x.values)

# making training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# fitting the SVC model using x_train and y_train
print("Fitting model")
svc = SVC()
svc.fit(x_train, y_train)

# predicting y_test using the model
print("Making predictions")
y_pred = svc.predict(x_test)

# finding accuracy of the model
accuracy = accuracy_score(y_pred, y_test)
print(f"Accuracy: {accuracy}")