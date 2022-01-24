import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

dataset = pd.read_excel("otu.xlsx")
test_size = .2
X = dataset.drop('Output', axis=1)
y = dataset['Output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

model = MultinomialNB().fit(X_train, y_train)


predicted = model.predict(X_test)
print(classification_report(y_test,predicted))

print("The test size is: {0}".format(test_size))
print("The prediction accuracy is: {:0.2f}%".format(np.mean(predicted == y_test) * 100))

