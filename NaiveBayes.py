import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_excel("otu.xlsx")

X = dataset.drop('Output', axis=1)
y = dataset['Output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model = MultinomialNB().fit(X_train, y_train)

predicted = model.predict(X_test)

print("The prediction accuracy is: {:0.2f}%".format(np.mean(predicted == y_test) * 100))
