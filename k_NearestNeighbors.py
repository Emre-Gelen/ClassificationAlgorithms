import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_excel("otu.xlsx")

X = dataset.drop('Output', axis=1)
y = dataset['Output']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.16)


scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))

score = classifier.score(X_test, y_test)
print("The prediction accuracy is: {:0.2f}%".format(score * 100))


