#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
np.random.seed(0)

# %%
# Prepare the data
n_classes = 2
data_x, data_y = [], []
for i in range(n_classes):
	data_x.append(np.random.gumbel(loc=i*4, size=(50, 2)) * 20)
	data_y.append(np.ones(50) * i)

for c in data_x:
	plt.scatter(c[:,0], c[:,1])
plt.show()

X = np.array(data_x).reshape(-1, 2)
y = np.array(data_y).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0.2, random_state=42)

#%%
# Create SVM
svc = SVC(kernel="poly")
svc.fit(X_train, y_train)

# %%
# Check accuracy
y_pred = svc.predict(X_test)
print(classification_report(y_test,y_pred))

# %%
# Plot support vectors
ax = plt.subplot(1, 1, 1)
for c in data_x:
	ax.scatter(c[:,0], c[:,1])
ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()

# %%
