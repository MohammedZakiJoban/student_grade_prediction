
import pandas as p1
import numpy as n1
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = p1.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = 'G3'


x = n1.array(data.drop([predict], 1))
y = n1.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
best = 0
"""
for _ in range(30):




    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#training the model

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test,y_test)

    if acc > best:
        best = acc
        with open("studentModel.pickle", "wb") as f:
            pickle.dump(linear, f)

#Open trained model
"""
pickle_in = open("studentModel.pickle", 'rb')
linear = pickle.load(pickle_in)






predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("final grade")
pyplot.show()

