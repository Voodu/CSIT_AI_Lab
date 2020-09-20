#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
# %%
# Load and prepare iris data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
# Train the kNN classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
# %%
# Check accuracy
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

#%% 
# Check prediction on random test data
sample_ix = int(np.random.rand() * len(X_test))
predicted = classifier.predict(np.expand_dims(X_test[sample_ix], 0))[0]
real = y_test[sample_ix]
print("Predicted:", predicted)
print("Real:", real)
print("Correct:", predicted == real)
