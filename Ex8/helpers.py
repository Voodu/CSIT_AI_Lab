# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from params import FuzzyInputVariable_3Trapezoids, FuzzyInputVariable_2Trapezoids, FuzzyInputVariable_List_Trapezoids
from operators import productN
import numpy as np
#from helps_and_enhancers import *
import matplotlib.pyplot as plt
from ANFIS import ANFIS
import time
import copy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# %% [markdown]
# <H1>Przygotowanie zbioru danych: XOR<H1>

# %%
data = np.random.randint(0, 6, (100, 2))
data_labels = np.prod(data, axis=1) 
# %% [markdown]
# <H1>Utworzenie funkcji przynależności</H1>

# %%
# varX = FuzzyInputVariable_2Trapezoids(0.5, 0.5, "XAxis", ["L","H"]) # low, high
# varY = FuzzyInputVariable_2Trapezoids(0.5, 0.5, "YAxis", ["L","H"])

mf1 = [[-0.5, 0.25, 0.25, 0.25], [0.5, 0.25, 0.1, 0.1]]
varX = FuzzyInputVariable_List_Trapezoids(mf1, "XAxis", ["L","H"])
varY = FuzzyInputVariable_List_Trapezoids(mf1, "YAxis", ["L","H"])

#Wyświetlanie funkcji przynależnosci
plt.figure()
varX.show()
plt.legend()

plt.figure()
varY.show()
plt.legend()

plt.show()

# %% [markdown]
# <H1>Inicjalizacja systemu ANFIS</H1>

# %%
X_train, X_test, y_train, y_test = train_test_split(data, data_labels, test_size=0.2, random_state=25)

fis = ANFIS([varX, varY], X_train.T, y_train)

print("Parametry początkowe:\nPrzesłanki: ",fis.premises, "\nKonkluzje: ", fis.tsk)

fis.show_results()

# %% [markdown]
# <H1>Uczenie systemu ANFIS</H1>

# %%
start = time.time()
fis.train(True, True, False, True, n_iter=75)
end = time.time()
print("TIME elapsed: ", end - start)   
fis.training_data = X_train.T
fis.expected_labels = y_train
fis.show_results()


# %%
fis.training_data = X_test.T
fis.expected_labels = y_test
fis.show_results()

y_pred = fis.anfis_estimate_labels(fis.premises,fis.op,fis.tsk)
y_pred = list(map(round,y_pred.flatten()))
print(confusion_matrix(y_test, y_pred))

# %% [markdown]
# <H1>Sprawdzanie parametrów jakich nauczył się system</H1>

# %%
#Wyświetlanie funkcji przynależnosci
plt.figure()
varX.show()
plt.legend()

plt.figure()
varY.show()
plt.legend()

plt.show()


# %%
print("Parametry końcowe:\nPrzesłanki: ",fis.premises, "\nKonkluzje: ", fis.tsk)

# %% [markdown]
# <H1>Sprawdzanie wpływu parametrów na wyniki systemu</H1>

# %%
fis.training_data = data.T
fis.expected_labels = data_labels
fis.show_results()


# %%
# fis.premises = ##################
fis.training_data = data.T
fis.expected_labels = data_labels
fis.show_results()


# %%
# fis.tsk = ##################
fis.training_data = data.T
fis.expected_labels = data_labels
fis.show_results()

