# %% 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data():
	# Loading the dataset
	dataset = pd.read_csv('iris.data')
	dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True) # Shuffle
	dataset = pd.get_dummies(dataset, columns=['class']) # One Hot Encoding
	values = list(dataset.columns.values)

	target = dataset[values[-3:]]
	target = np.array(target, dtype='float32')
	X = dataset[values[:-3]]
	X = np.array(X, dtype='float32')

	# Creating a Train and a Test Dataset
	test_size =  int(len(dataset) * 0.8)
	train_x = X[:test_size]
	test_x = X[test_size:]
	train_y = target[:test_size]
	test_y = target[test_size:]

	return train_x, test_x, train_y, test_y
 
np.random.seed(0)
x_train, test_x, y_train, test_y = load_data()

# %%
class NN:
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.net = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []

    def relu(self, inputs):
       return np.maximum(0, inputs)

    def relu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
 
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)
 
    def init_network(self):
        np.random.seed(1)
        for l in range(1, len(self.layers_size)):
            self.net[f"W{l}"] = np.random.randn(self.layers_size[l], self.layers_size[l - 1])
            self.net[f"b{l}"] = np.zeros((self.layers_size[l], 1))
 
    def forward(self, X):
        params = {}
 
        H = X.T
        for l in range(self.L - 1):
            Z = self.net[f"W{l+1}"].dot(H) + self.net[f"b{l+1}"]
            H = self.relu(Z)
            params[f"H{l+1}"] = H
            params[f"W{l+1}"] = self.net[f"W{l+1}"]
            params[f"Z{l+1}"] = Z
 
        Z = self.net[f"W{self.L}"].dot(H) + self.net[f"b{self.L}"]
        H = self.softmax(Z)
        params[f"H{self.L}"] = H
        params[f"W{self.L}"] = self.net[f"W{self.L}"]
        params[f"Z{self.L}"] = Z
 
        return H, params
 
    def backward(self, X, Y, params):
 
        derivatives = {}
 
        params["H0"] = X.T # just input, but in the standarized way
 
        H = params[f"H{self.L}"] # network output
        dZ = H - Y.T # CE prime
        # Taking random part of H and Y will make MB GD?
 
        dW = dZ.dot(params[f"H{self.L - 1}"].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dHPrev = params[f"W{self.L}"].T.dot(dZ)
 
        derivatives[f"dW{self.L}"] = dW
        derivatives[f"db{self.L}"] = db
 
        for l in range(self.L - 1, 0, -1):
            dZ = dHPrev * self.relu_derivative(params[f"Z{l}"])
            dW = dZ.dot(params[f"H{l-1}"].T) / self.n
            db = np.sum(dZ, axis=1, keepdims=True) / self.n
            if l > 1:
                dHPrev = params[f"W{l}"].T.dot(dZ)
 
            derivatives[f"dW{l}"] = dW
            derivatives[f"db{l}"] = db

        return derivatives
 
    def fit(self, X, Y, l_rate=0.01, n_iterations=2500):
        np.random.seed(1)
        self.n = X.shape[0] # number of samples
        self.layers_size.insert(0, X.shape[1]) # define size of the 1st layer
        self.init_network()
        for i in range(n_iterations):
            H, params = self.forward(X)
            derivatives = self.backward(X, Y, params)
 
            for l in range(1, self.L + 1):
                self.net[f"W{l}"] -= l_rate * derivatives[f"dW{l}"]
                self.net[f"b{l}"] -= l_rate * derivatives[f"db{l}"]
 
            if i % 10 == 0:
                cost = -np.mean(Y * np.log(H.T+ 1e-8))
                self.costs.append(cost)
 
    def predict(self, input, target):
        H, _ = self.forward(input)
        predicted = np.argmax(H, axis=0)
        target = np.argmax(target, axis=1)
        accuracy = (predicted == target).mean()
        return accuracy * 100
 
    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs / 10")
        plt.ylabel("cost")
        plt.show()
 
 

 
 
#%%

layers = [10, 12, 3]

nn = NN(layers)
nn.fit(x_train[:], y_train[:], l_rate=0.1, n_iterations=1000)
print("Train Accuracy:", nn.predict(x_train[:], y_train[:]))
print("Test Accuracy:", nn.predict(test_x, test_y))
nn.plot_cost()
# %%
