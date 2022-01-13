import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

dataset = pd.read_excel("otu.xlsx")

X = dataset.drop('Output', axis=1)
y = dataset['Output']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(classification_report(y_test,y_pred))

score = svclassifier.score(X_test, y_test)
print("The prediction of linear kernel accuracy is: {:0.2f}%".format(score * 100))