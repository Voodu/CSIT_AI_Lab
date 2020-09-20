#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
np.random.seed(0)

# %%
# Prepare the data
n_points = 100
noise = 60
data = np.arange(n_points*2).reshape(-1, 2)
data = data + np.random.rand(data.shape[0], 2) * noise * 2 + (-noise)
plt.scatter(data[:,0], data[:,1])
plt.show()

X_train, X_test, y_train, y_test = train_test_split(data[:,0], data[:,1], test_size=0.2, random_state=42)


#%% 
# Create and train regression model
# regr = linear_model.LinearRegression()
regr = linear_model.LinearRegression()
regr.fit(X_train.reshape(-1, 1), y_train)

#%%
# Make predictions using the testing set and plot it
y_pred = regr.predict(X_test.reshape(-1, 1))
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue')
plt.show()