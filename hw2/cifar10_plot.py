"""
Created by haiphung106
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from keras.callbacks import History, ModelCheckpoint
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.regularizers import l2
from keras.utils import to_categorical
import os
import cv2
from layer import *

"""
Loading and pre-process data
"""
path = 'cifar-10-python'
num_train_samples = 50000
num_test_samples = 10000
size_figure = 32 * 32 * 3
train_data = []
train_label = []

for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (train_data_temp, train_label_temp) = load_cifar10_batch(fpath)
    train_data.append(train_data_temp)
    train_label.append(train_label_temp)
train_label = np.reshape(train_label, (-1, 1))

fpath = os.path.join(path, 'test_batch')
test_data, test_label = load_cifar10_batch(fpath)
train_data = np.reshape(train_data, (num_train_samples, size_figure))
test_data = np.reshape(test_data, (num_test_samples, size_figure))
num_classes = 10
train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=0.1, shuffle=True)
ground_truth_label = test_label
ground_truth_label = np.reshape(ground_truth_label, (-1, 1))
train_label = to_categorical(train_label, num_classes)
test_label = to_categorical(test_label, num_classes)
val_label = to_categorical(val_label, num_classes)

"""
Loading model
"""

model = load_model('cifar10_normal_model.h5')
model_weight = model.get_weights()
model_reg = load_model('cifar10_regularizer_model.h5')
model_weight_reg = model_reg.get_weights()

"""
Reshape and normalize
"""
train_data = train_data.reshape(train_data.shape[0], 3, 32, 32).astype('float32')
val_data = val_data.reshape(val_data.shape[0], 3, 32, 32).astype('float32')
test_data = test_data.reshape(test_data.shape[0], 3, 32, 32).astype('float32')
train_data = train_data.transpose(0, 2, 3, 1)
val_data = val_data.transpose(0, 2, 3, 1)
test_data = test_data.transpose(0, 2, 3, 1)
input_shape = (32, 32, 3)
train_data /= 255
val_data /= 255
test_data /= 255

"""
Plot history layer
"""
show_histogram_of_weight = True
if show_histogram_of_weight == True:
    # layer 1: convolution
    weight_conv1 = model_weight_reg[0]
    weight_conv1 = np.reshape(weight_conv1, (-1, 1))
    plt.hist(weight_conv1, bins=40)
    plt.title('Histogram of Weight - Conv1')
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.show()

    # layer 2: convolution
    weight_conv2 = model_weight_reg[2]
    weight_conv2 = np.reshape(weight_conv2, (-1, 1))
    plt.hist(weight_conv2, bins=40)
    plt.title('Histogram of Weight - Conv2')
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.show()

    # layer 4: convolution
    weight_conv3 = model_weight_reg[4]
    weight_conv3 = np.reshape(weight_conv3, (-1, 1))
    plt.hist(weight_conv3, bins=80)
    plt.title('Histogram of Weight - Conv3')
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.show()

    # layer 7: Dense (Fully connecteced)
    weight_fc1 = model_weight_reg[6]
    weight_fc1 = np.reshape(weight_fc1, (-1, 1))
    plt.hist(weight_fc1, bins=80)
    plt.title('Histogram of Weight - Dense1')
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.show()

    # layer 8: Dense (Fully connected)
    weight_fc2 = model_weight_reg[8]
    weight_fc2 = np.reshape(weight_fc2, (-1, 1))
    plt.hist(weight_fc2, bins=90)
    plt.title('Histogram of Weight - Dense2')
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()

    # layer 9: Dense (Fully connected)
    weight_fc3 = model_weight_reg[10]
    weight_fc3 = np.reshape(weight_fc3, (-1, 1))
    plt.hist(weight_fc3, bins=90)
    plt.title('Histogram of Weight - Dense3')
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()

"""
Classify correct and incorrect class
"""
show_incorrect_image = True
if show_incorrect_image == True:
    predicted_classes = model.predict_classes(test_data)
    correct_classes = np.argmax(test_label, axis=1)
    correct_indices = np.nonzero(predicted_classes == correct_classes)[0]
    incorrect_indices = np.nonzero(predicted_classes != correct_classes)[0]

    incorrect_picture = test_data[incorrect_indices]
    incorrect_picture = incorrect_picture[:, :, :, :]
    correct_picture = test_data[correct_indices]
    correct_picture = correct_picture[:, :, :, :]
    incorrect_label = predicted_classes[incorrect_indices]
    correct_label = ground_truth_label[incorrect_indices]
    amount_incorrect_picture = len(incorrect_indices)
    amount_correct_picture = len(correct_indices)
    """
    Classify to compare correct and incorrect pictures and class
    """
    name_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # Incorrect class
    for i in range(amount_incorrect_picture):
        img = incorrect_picture[i][:,:,:]
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Predict: {}'.format(str(name_classes[incorrect_label[i]])) +  'Label: {}'.format(str(name_classes[int(correct_label[i])])))
        plt.savefig('cifar10_incorrect_classification/' + str(i) + '.png', bbox_inches='tight')
        plt.close(fig)

    # Correct class
    for i in range(amount_correct_picture):
        img = correct_picture[i][:,:,:]
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Predict: {}'.format(str(name_classes[correct_label[i]])) + 'Label: {}'.format(str(name_classes[correct_label[i]])))
        plt.savefig('cifar10_correct_classification/' + str(i) +  '.png', bbox_inches='tight')
        plt.close(fig)

    print(len(correct_indices), 'Classified Correctly')
    print(len(incorrect_indices), 'Classified Incorrectly')

    """
    Plot features map
    """
    md1 = Model(input=model.inputs, output=model.layers[0].output)
    md2 = Model(input=model.inputs, output=model.layers[1].output)
    md3 = Model(input=model.inputs, output=model.layers[2].output)
    md4 = Model(input=model.inputs, output=model.layers[3].output)
    md5 = Model(input=model.inputs, output=model.layers[4].output)

    for idx in [26]:
        out_md1 = md1.predict(correct_picture[idx].reshape(1, 32, 32, 3))[0, :, :, :]
        out_md2 = md2.predict(correct_picture[idx].reshape(1, 32, 32, 3))[0, :, :, :]
        out_md3 = md3.predict(correct_picture[idx].reshape(1, 32, 32, 3))[0, :, :, :]
        out_md4 = md4.predict(correct_picture[idx].reshape(1, 32, 32, 3))[0, :, :, :]
        out_md5 = md5.predict(correct_picture[idx].reshape(1, 32, 32, 3))[0, :, :, :]

    fig = plt.figure(figsize=(8, 6))

    plt.subplot(231)
    plt.imshow(out_md1[:, :, 0])
    plt.title('Cov Layer 1')

    plt.subplot(232)
    plt.imshow(out_md2[:, :, 0])
    plt.title('Cov Layer 2')

    plt.subplot(233)
    plt.imshow(out_md3[:, :, 0])
    plt.title('Maxpooling Layer 1')

    plt.subplot(234)
    plt.imshow(out_md4[:, :, 0])
    plt.title('Cov Layer 3')

    plt.subplot(235)
    plt.imshow(out_md5[:, :, 0])
    plt.title('Maxpooling Layer 2')
    plt.show()
