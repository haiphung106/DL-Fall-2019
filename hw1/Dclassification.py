"""
Created by haiphung106
"""

import numpy as np
import matplotlib.pyplot as plt
from function import *
import csv
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

d = []
features = 35
sizes = 0
with open('ionosphere_csv.csv') as csv_file:
    file = csv.reader(csv_file, delimiter=',')
    index = 0
    for row in file:
        if (index == 0):
            index += 1
            continue
        for i in range(features - 1):
            d.append(float(row[i]))
        if (row[features - 1] == 'g'):
            d.append(1)
        else:
            d.append(0)

        sizes += 1
d = np.reshape(d, (sizes, features))

sizes = len(d)
x = d[:, 0:34]
t_noncovert = d[:, 34]
t_noncovert = np.reshape(t_noncovert, (sizes, 1))
t = []
for i in t_noncovert:
    t.append(convertBinary(i))
t = np.reshape(t, (sizes, 2))

train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=0.2, shuffle=True)
train_size = train_x.shape[0]
test_size = test_x.shape[0]
LR = 0.005
batch_size = 32
epochs = 2000
every = 100
activation = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'softmax']
NN = NeuralNetwork([34, 128, 64, 32, 3, 2], activation=activation, LR=LR)
error_rate_train = []
error_rate_test = 0
entropy_error_train = []
hidden_good_10 = []
hidden_bad_10 = []
hidden_good_390 = []
hidden_bad_390 = []
hidden10 = []
hidden390 = []

time = train_size // batch_size
NN.init_weight()
for e in range(epochs):
    epochs_error_number = 0
    epochs_entropy_error = 0
    for i in range(time):
        input = train_x[i * batch_size:(i + 1) * batch_size]
        output = train_t[i * batch_size:(i + 1) * batch_size]
        NN.feedforward(input)
        NN.backpropagation_classification(input, output)
        NN.weight_update()
        epochs_error_number += count_diff(output, NN.predict_y(input).round())
        epochs_entropy_error += NN.cross_entropy(input, output)
    error_rate_train.append(round((epochs_error_number / train_size) * 100, 2))
    entropy_error_train.append(epochs_entropy_error / train_size)
    if e == 10:
        NN.feedforward(train_x)
        hidden10 = NN.t[NN.number_layer - 1]
    if e == 390:
        NN.feedforward(train_x)
        hidden390 = NN.t[NN.number_layer - 1]

for i in range(train_size):
    if train_t[i][0] == 1:
        hidden_good_10.append(hidden10[i])
    else:
        hidden_bad_10.append(hidden10[i])

for i in range(train_size):
    if train_t[i][0] == 1:
        hidden_good_390.append(hidden390[i])
    else:
        hidden_bad_390.append(hidden390[i])

hidden_good_10 = np.array(hidden_good_10)
hidden_bad_10 = np.array(hidden_bad_10)
hidden_good_390 = np.array(hidden_good_390)
hidden_bad_390 = np.array(hidden_bad_390)

data_error = count_diff(test_t, NN.predict_y(test_x).round())
error_rate = data_error / test_size
error_rate_test = round(error_rate * 100, 2)


print('Train Prediction: {}%'.format(error_rate_train[-1]))
print('Test Prediction: {}%'.format(error_rate_test))

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot(error_rate_train, 'b-')
plt.title('Error Rate Training Phase')
plt.xlabel('#epoch case')
plt.ylabel('Error rate')

plt.subplot(122)
plt.plot(entropy_error_train, 'b-')
plt.xlabel('#epoch case')
plt.title('Cross Entropy Training Phase')
plt.ylabel('Cross Entropy Error')

# plot 2D
plt.figure(figsize=(4, 4))
plt.subplot(121)
plt.plot(hidden_good_10[:, 0], hidden_good_10[:, 1], 'go', label='Good')
plt.plot(hidden_bad_10[:, 0], hidden_bad_10[:, 1], 'ro', label='Good')
plt.title("2D feature 10th epoch")
plt.subplot(122)
plt.plot(hidden_good_390[:, 0], hidden_good_390[:, 1], "go", label="Good")
plt.plot(hidden_bad_390[:, 0], hidden_bad_390[:, 1], "ro", label="Bad")
plt.title("2D feature 390th epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# plot 3D
ax = plt.subplot(121, projection='3d')
ax.scatter(hidden_good_10[:, 0], hidden_good_10[:, 1], hidden_good_10[:, 2], marker='o', color="green", label="Good",
           alpha=1.0)
ax.scatter(hidden_bad_10[:, 0], hidden_bad_10[:, 1], hidden_bad_10[:, 2], marker='o', color="red", label="Bad",
           alpha=1.0)
plt.title("3D feature 10th epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax = plt.subplot(122, projection='3d')
ax.scatter(hidden_good_390[:, 0], hidden_good_390[:, 1], hidden_good_390[:, 2], marker='o', color="green", alpha=1.0,
           label="Good")
ax.scatter(hidden_bad_390[:, 0], hidden_bad_390[:, 1], hidden_bad_390[:, 2], marker='o', color="red", alpha=1.0,
           label="Bad")
plt.title("3D feature 390th epoch")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
