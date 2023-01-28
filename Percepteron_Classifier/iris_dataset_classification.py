"Last Update: 1/28/2023"

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Iris dataset URL or Local address of a csv format dataset
#s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
s = 'iris.data'

print('From URL:', s)

# Reading the csv file to a dataframe named df
df = pd.read_csv(s, header=None, encoding="utf-8")
# print the last 5 rows od the dataframe
print(df.tail())

# rows 0-49 -> Iris-setosa : labeled as 0
# rows 50-99 -> Iris-versicolor : labeled as 1
# rows 100-149 -> Iris-virginica : labeled as 2
# We only want setosa and versicolor
y = df.iloc[0:150,4].values
y = np.where(y == "Iris-setosa", 0, y)
y = np.where(y == "Iris-versicolor", 1, y)
y = np.where(y == "Iris-virginica", 2, y)

# Select the sepal length (first column) and petal length (third column)
X = df.iloc[0:100, [0, 2]].values

# Visualize the data
plt.figure()
plt.scatter(X[:50, 0], X[:50, 1], marker='o', color='blue', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], marker='*', color='red', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='lower right')
# plt.show()


from Percepteron_Classifier import Percepteron

p = Percepteron(lr=0.1, n_iter=10)
p.fit(X, y)

plt.figure()
plt.plot(range(1, len(p.error_) + 1), p.error_, marker='o')
# plt.show()



from matplotlib.colors import ListedColormap
cmap = ListedColormap(['blue', 'red'])

resolution=0.02
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
mesh_labels = p.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
mesh_labels = mesh_labels.reshape(xx1.shape)

plt.figure()
plt.contourf(xx1, xx2, mesh_labels, alpha=0.25, cmap=cmap)
plt.scatter(X[:50, 0], X[:50, 1], marker='o', color='blue', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], marker='*', color='red', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='lower right')
plt.show()

