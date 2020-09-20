#%%
from matplotlib.pyplot import imshow
import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
# %%
# Load and prepare the data
CIFAR = True
if CIFAR:
	(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
	train_labels, test_labels = train_labels.flatten(), test_labels.flatten()
	class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	img_shape=(32, 32, 3)
else:
	(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
	train_images,test_images = np.expand_dims(train_images, 3), np.expand_dims(test_images, 3)
	class_names = ['0','1','2','3','4','5','6','7','8','9']
	img_shape=(28, 28, 1)

train_images, test_images = train_images / 255.0, test_images / 255.0 # scale to 0.0 - 1.0

# %%
# Declare model layers. Conv to extract features, dense to classify
model = models.Sequential([
		layers.Conv2D(32, (3, 3), activation='relu', input_shape=img_shape),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(64, (3, 3), activation='relu'),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(64, (3, 3), activation='relu'),
		layers.Flatten(),
		layers.Dense(64, activation='relu'),
		layers.Dense(10),
	])
model.summary()

# %%
# Pick optimizer, loss, metrics & train
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# %%
# Pick some specific test image, make prediction and compare with actual label
img_ix = int(np.random.rand() * len(test_images))
img = test_images[img_ix]
img = (np.expand_dims(img,0)) # i.e. (28,28) -> (1, 28, 28) to match required shape

real = class_names[test_labels[img_ix]]
prediction = class_names[np.argmax(model.predict(img))]

plt.figure()
if CIFAR:
	plt.imshow(img[0])
else:
	plt.imshow(img[0].reshape((28,28)))
plt.xticks([])
plt.yticks([])
plt.xlabel(f"real: {real}" + "\n" + f"pred: {prediction}" )
plt.show()

# %%
