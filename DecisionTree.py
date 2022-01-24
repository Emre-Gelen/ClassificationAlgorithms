import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree

dataset = pd.read_excel("otu.xlsx")
test_size = .15
max_depth = 4
X = dataset.drop('Output', axis=1)
y = dataset['Output']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

classifier = DecisionTreeClassifier(max_depth=max_depth)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))

score = classifier.score(X_test, y_test)
print("The test size is: {0}".format(test_size))
print("The maximum depth is: {0}".format(max_depth))
print("The prediction accuracy is: {:0.2f}%".format(score * 100))

text_representation = tree.export_text(classifier)
print(text_representation)