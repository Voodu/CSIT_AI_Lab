# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

class Activations:
    def relu(inputs):
        return np.maximum(0, inputs)
    
    def dRelu(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

#%%
import pandas as pd

np.random.seed(0)

# Loading the dataset
dataset = pd.read_csv("iris.data")
dataset = dataset.sample(frac=1, random_state=1).reset_index(drop=True) # Shuffle
dataset = pd.get_dummies(dataset, columns=["class"]) # One Hot Encoding
values = list(dataset.columns.values)

target = dataset[values[-3:]]
target = np.array(target, dtype="float32")
X = dataset[values[:-3]]
X = np.array(X, dtype="float32")

# Creating a Train and a Test Dataset
test_size =  int(len(dataset) * 0.8)
X_train = X[:test_size]
X_test = X[test_size:]
y_train = target[:test_size]
y_test = target[test_size:]

#%%
class NN:
    def __init__(self, neurons1, neurons2, learning_rate, epochs):
        self.num_neurons = [neurons1, neurons2]
        self.network = {}
        self.network["w1"] = 0.10 * np.random.randn(4, neurons1)
        self.network["w2"] = 0.10 * np.random.randn(neurons1, neurons2)
        self.network["w3"] = 0.10 * np.random.randn(neurons2, 3)
        self.network["b1"] = np.zeros((1, neurons1))
        self.network["b2"] = np.zeros((1, neurons2))
        self.network["b3"] = np.zeros((1, 3))

        self.params = {}
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = []

    def forward(self):
        h = self.input
        activation = [None, Activations.relu, Activations.relu, Activations.softmax]
        
        for i in range(1, 4):
            z = h.dot(self.network[f"w{i}"]) + self.network[f"b{i}"]
            h = activation[i](z)
            self.params[f"z{i}"] = z
            self.params[f"h{i}"] = h

        output = self.params["h3"]
        loss = self.loss_function(self.target, output)
        return output, loss

    def loss_function(self, target, predicted):
        return -np.sum(target * np.log(predicted))

    def loss_function_prime(self, target, predicted):
        return predicted - target

    def backward(self, predicted):
        n = self.input.shape[0]
        dvs = {} #derivatives
        self.params["h0"] = self.input

        dl_z = predicted - self.target # NNL wrt output; https://www.labri.fr/perso/vlepetit/teaching/selected_topics_on_computer_vision_material/ranzato_cvpr2014_DLtutorial.pdf
        dvs["w3"] = np.dot(self.params["h2"].T, dl_z) / n
        dvs["b3"] = np.sum(dl_z, axis=0) / n
        dl_h = np.dot(dl_z, self.network["w3"].T)

        for i in range(2, 0, -1): #loop for 2, 1
            dl_z = dl_h * Activations.dRelu(self.params[f"z{i}"])
            dvs[f"w{i}"] = np.dot(self.params[f"h{i-1}"].T, dl_z) / n
            dvs[f"b{i}"] = np.sum(dl_z, axis=0) / n
            if i > 1:
                dl_h = np.dot(dl_z, self.network[f"w{i}"].T)

        #update the weights and bias
        for i in range(1, 4):
            self.network[f"w{i}"] -= self.learning_rate * dvs[f"w{i}"]
            self.network[f"b{i}"] -= self.learning_rate * dvs[f"b{i}"]


    def fit(self, input, target):
        self.input = input
        self.target = target

        for i in range(self.epochs):
            predicted, loss = self.forward()
            self.backward(predicted)
            self.loss.append(loss)
        
    def predict(self, X):
        h1 = Activations.relu(X.dot(self.weights["l1"]) + self.biases["l1"])
        h2 = Activations.relu(h1.dot(self.weights["l2"]) + self.biases["l2"])
        out = Activations.softmax(h2.dot(self.weights["l3"]) + self.biases["l3"])
        return np.round(out)

    def acc(self, y, yhat):
        acc = 0
        print(y, yhat)
        return acc

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("Cross Entropy loss")
        plt.title("Loss curve for training")
        plt.show()


#%%
np.random.seed(0)
nn = NN(15, 15, 0.01, 1000)
nn.fit(X_train, y_train)
nn.plot_loss()

#%%
train_pred = nn.predict(X_train)
test_pred = nn.predict(X_test)

print("Train accuracy is {}".format(nn.acc(y_train, train_pred)))
print("Test accuracy is {}".format(nn.acc(y_test, test_pred)))