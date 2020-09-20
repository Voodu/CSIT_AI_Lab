#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate and show data
clusters = 4
X, y = make_blobs(centers=clusters, random_state=1, n_samples=200)
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
# %%
# Declare clustering options and fit it to data
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
k_centers = kmeans.cluster_centers_

# %%
# Show the data and centroids
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y)
plt.scatter(k_centers[:,0],k_centers[:,1])
for ix, tp in enumerate(k_centers):
	plt.annotate(f"K{ix}", tp)
plt.show()

#%%
# Test prediction on several random points
x_test = np.random.rand(10)*(np.max(X[:, 0]) - np.min(X[:, 0])) + np.min(X[:, 0])
y_test = np.random.rand(10)*(np.max(X[:, 1]) - np.min(X[:, 1])) + np.min(X[:, 1])
test_points = np.concatenate((x_test, y_test)).reshape(10, 2)
predicted = kmeans.predict(test_points)
ax = plt.subplot(1, 1, 1)
ax.scatter(X[:,0], X[:,1], c=y)
ax.scatter(k_centers[:,0],k_centers[:,1])
for ix, tp in enumerate(k_centers):
	ax.annotate(f"K{ix}", tp)
for ix, tp in enumerate(test_points):
	ax.plot(tp[0], tp[1], "or", color="blue")
	ax.annotate(f"P{predicted[ix]}", tp)


# %%
