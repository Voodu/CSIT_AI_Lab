#%%
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load data
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
class_names = ['0','1','2','3','4','5','6','7','8','9']

# %%
# Scale to 0.0-1.0 range
train_images = train_images / 255.0
test_images = test_images / 255.0
# %%
# Declare network layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # changes 2D to 1D array
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Softmax()
])
# %%
# Specify optimizer, loss, metrics & compile it
model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# %%
# Train the network
model.fit(train_images, train_labels, epochs=10)
# %%
# Pick some specific test image
img_ix = int(np.random.rand() * len(test_images))
img = test_images[img_ix]
img = (np.expand_dims(img,0)) # i.e. (28,28) -> (1, 28, 28) to match required shape

# %%
# Display it
plt.figure()
plt.imshow(img[0])
plt.xticks([])
plt.yticks([])
plt.xlabel(class_names[test_labels[img_ix]])
plt.show()
# %%
# Make a prediction using the model
predictions_single = model.predict(img)
np.argmax(predictions_single[0])