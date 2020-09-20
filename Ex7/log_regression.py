#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

np.random.seed(3)

# %%
# Prepare the data
n_classes = 5
points_per_class = 30
data_x, data_y = [], []
for i in range(n_classes):
	data_x.append(np.random.gumbel(loc=i*4, size=(points_per_class, 2)) * 20)
	data_y.append(np.ones(points_per_class) * i)
indexes = np.random.randint(0, n_classes, n_classes//2)
for i in indexes: # add some noise to the data
	data_x[i][:, 1] += np.random.randint(-200, 200)
for c in data_x:
	plt.scatter(c[:,0], c[:,1])
plt.show()

X = np.array(data_x).reshape(-1, 2)
Y = np.array(data_y).reshape(-1, 1)
# %%
# Create and train regression model
logreg = linear_model.LogisticRegression()
logreg.fit(X, Y)

# %%
# Plot the detected regions
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = 10  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('X')
plt.ylabel('Y')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
# %%
