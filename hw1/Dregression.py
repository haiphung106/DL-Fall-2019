"""
Created by haiphung106
"""

import numpy as np
import matplotlib.pyplot as plt
from function import *
import csv
from sklearn.model_selection import train_test_split

data = []
size = 0
features = 10
with open('EnergyEfficiency_data.csv') as csv_file:
    file = csv.reader(csv_file, delimiter=',')
    index = 0
    for row in file:
        if (index == 0):
            index += 1
            continue
        for i in range(int(features)):
            data.append(float(row[i]))
        size += 1

data = np.reshape(data, (size, features))
size = len(data)

categorical_1 = np.reshape(data[:, 5], (size, 1))
categorical_2 = np.reshape(data[:, 7], (size, 1))

categorical_1 = Onehotvector(categorical_1)
categorical_2 = Onehotvector(categorical_2)
nfeatures = np.column_stack((data[:, 0:5], data[:, 6], categorical_1, categorical_2))

x = np.column_stack((data[:, 0:5], data[:, 6]))
t = np.array(data[:, 8].astype(float)).reshape(-1, 1)

for i in range(6):
    nfeatures[:, i] /= nfeatures[:, i].max(axis=0)

nfeatures = np.array(nfeatures, dtype=float)

train_x, test_x, train_t, test_t = train_test_split(nfeatures, t, test_size=0.25, shuffle=False)

no_remove_feature = 1
no_select_feature = nfeatures - no_remove_feature
select_index = []
for i in range(len(nfeatures[0])):
    select_index.append(i)
std = []
for i in range(6):
    std.append(np.std(nfeatures[:, i]))
for i in range(no_remove_feature):
    min_std = max(std)
    remove_index = std.index(min_std)
    select_index.remove(remove_index)
    std.remove(min_std)
if no_remove_feature == 0:
    print('Selection All Feature')
else:
    # print('Number of Selected Feature' + str(no_select_feature))
    print('Selected Feature Index' + str(select_index))

train_size = train_x.shape[0]
test_size = test_x.shape[0]
LR = 0.002
batch_size = 32
epochs = 10000
activation = ['sigmoid', 'sigmoid', 'linear']
NN = NeuralNetwork([16, 10, 10, 1], activation=activation, LR=0.002)
print(NN.__repr__())
time = train_size // batch_size
every = 1000
model = []
error = []
ERMS = []
NN.init_weight()
for e in range(epochs):
    epochs_error = 0
    for i in range(time):
        input = train_x[i * batch_size:(i + 1) * batch_size]
        output = train_t[i * batch_size:(i + 1) * batch_size]
        NN.feedforward(input)
        NN.backpropagation(input, output)
        NN.weight_update()
    t_predict = NN.predict(train_x)

    epochs_error = np.sum((t_predict - train_t) ** 2)
    error.append(epochs_error)
    ERMS = np.sqrt(epochs_error / train_size)
    if (e % every == 0):
        print('Epoch {}, ERMS = {}'.format(e, float(epochs_error), float(ERMS)))

ERMS_train = ERMS
t_test_predict = NN.predict(test_x)
ERMS_test = np.sqrt(np.sum((t_test_predict - test_t) ** 2) / test_size)
print('Error training: {}'.format(ERMS_train))
print('Error testing: {}'.format(ERMS_test))
loss_error = np.reshape(error, (-1, 1))

plt.figure(figsize=(8, 4))
plt.plot(loss_error, 'g', linewidth=1, label='Target')
plt.title('Training Curve')
plt.xlabel('Epochs')
plt.ylabel('Sum of square error')

t_predict = NN.predict(train_x)
plt.figure(figsize=(8, 4))
plt.plot(t_predict, 'g', linewidth=1, label='Predict')
plt.plot(train_t, 'r', linewidth=1, label='Target')
plt.title('Prediction for Training Data')
plt.xlabel('#th case')
plt.ylabel('Heating Load')

plt.figure(figsize=(8, 4))
plt.plot(t_test_predict, 'g', linewidth=1, label='Predict')
plt.plot(test_t, 'r', linewidth=1, label='Target')
plt.title('Prediction for Testing Data')
plt.xlabel('#th case')
plt.ylabel('Heating Load')
plt.show()
