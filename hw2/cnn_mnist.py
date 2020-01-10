"""
Created by haiphung106
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import History, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.regularizers import l2
from keras.utils import to_categorical
from layer import *

(train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()
train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=1 / 12, shuffle=True)
train_label = to_categorical(train_label, 10)
test_label = to_categorical(test_label, 10)
val_label = to_categorical(val_label, 10)

"""
Reshape and normalize
"""
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32')
val_data = val_data.reshape(val_data.shape[0], 28, 28, 1).astype('float32')
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32')
input_shape = (28, 28, 1)
train_data /= 255
val_data /= 255
test_data /= 255

"""
Building CNN
"""
# reg_w = 0
reg_w = 0.01
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, kernel_regularizer=l2(reg_w), bias_regularizer=l2(reg_w)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), kernel_regularizer=l2(reg_w), bias_regularizer=l2(reg_w)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add((Flatten()))
model.add(Dense(256, activation=tf.nn.relu, kernel_regularizer=l2(reg_w)))  # , bias_regularizer=l2(reg_w)
model.add(Dense(10, activation=tf.nn.softmax, kernel_regularizer=l2(reg_w)))  # , bias_regularizer=l2(reg_w)

"""
Compile and Fitting
"""

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('regularizer_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit(x=train_data, y=train_label, epochs=20, validation_data=(val_data, val_label), batch_size=128,
                    verbose=2)  # ,callbacks=[model_checkpoint]

"""
Evaluate model
"""

_, training_accuracy = model.evaluate(train_data, train_label)
_, validation_accuracy = model.evaluate(val_data, val_label)

"""
Plot
"""
print('Training Accuracy: {}, Validation Accuracy: {}'.format(training_accuracy, validation_accuracy))
fig = plt.figure()
plt.subplot(121)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.title('Training Accuracy')
plt.ylabel('Accuracy rate')
plt.xlabel('Iteration')

plt.subplot(122)
plt.plot(history.history['loss'], label='Cross entropy')
plt.legend()
plt.title('Learning Curve')
plt.ylabel('Loss')
plt.xlabel('Iteration')

plt.show()
