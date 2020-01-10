"""
Created by haiphung106
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import History, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import load_model
from keras.utils import to_categorical
import cv2
from layer import *

"""
Load model
"""
model = load_model('normal_model.h5')
model_weight = model.get_weights()
model_reg = load_model('regularizer_model.h5')
model_weight_reg = model_reg.get_weights()

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
Plot history layer
"""
show_histogram_of_weight = True
if show_histogram_of_weight == True:
    # layer 1: convolution
    weight_conv1 = model_weight[0]
    weight_conv1 = np.reshape(weight_conv1, (-1, 1))
    plt.hist(weight_conv1, bins=40)
    plt.title('Histogram of Weight - Conv1')
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.show()

    # layer 2: convolution
    weight_conv2 = model_weight[2]
    weight_conv2 = np.reshape(weight_conv2, (-1, 1))
    plt.hist(weight_conv2, bins=40)
    plt.title('Histogram of Weight - Conv2')
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.show()

    # layer 6: Dense (Fully connecteced)
    weight_fc1 = model_weight[4]
    weight_fc1 = np.reshape(weight_fc1, (-1, 1))
    plt.hist(weight_fc1, bins=90)
    plt.title('Histogram of Weight - Dense1')
    plt.xlabel('Number')
    plt.ylabel('Value')
    plt.show()

    # layer 7: Dense (Fully connected)
    weight_fc2 = model_weight[6]
    weight_fc2 = np.reshape(weight_fc2, (-1, 1))
    plt.hist(weight_fc2, bins=80)
    plt.title('Histogram of Weight - Dense2')
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
    correct_label = predicted_classes[correct_indices]

    amount_incorrect_picture = len(incorrect_indices)
    amount_correct_picture = len(correct_indices)


    """
    Classify to compare correct and incorrect pictures and class
    """
    # Correct class
    for i in range(amount_correct_picture):
        img = correct_picture[i][:,:,0]
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Predict: '.format(str(correct_label[i])) + 'Label: '.format(str(correct_label[i])))
        plt.savefig('correct_classification/' + str(i) + '.png', bbox_inches='tight')
        plt.close(fig)
        # plt.show()


    # Incorrect class
    for i in range(amount_incorrect_picture):
        img = incorrect_picture[i]
        fig, ax = plt.subplot()
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Predict: '.format(str(incorrect_label[i])) + 'Label: '.format(str(correct_label[i])))
        plt.savefig('incorrect_classification/' + str(i) + '.png', bbox_inches='tight')
        plt.close(fig)
        # plt.show()

    print(len(correct_indices), 'Classified Correctly')
    print(len(incorrect_indices), 'Classified Incorrectly')

    """
    Feature maps
    """

    md1 = Model(input=model.inputs, output=model.layers[0].output)
    md2 = Model(input=model.inputs, output=model.layers[1].output)
    md3 = Model(input=model.inputs, output=model.layers[2].output)
    md4 = Model(input=model.inputs, output=model.layers[3].output)

    for idx in [16]:
        out_md1 = md1.predict(correct_picture[idx].reshape(1, 28, 28, 1))[0, :, :, :]
        out_md2 = md2.predict(correct_picture[idx].reshape(1, 28, 28, 1))[0, :, :, :]
        out_md3 = md3.predict(correct_picture[idx].reshape(1, 28, 28, 1))[0, :, :, :]
        out_md4 = md4.predict(correct_picture[idx].reshape(1, 28, 28, 1))[0, :, :, :]

    fig = plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.imshow(out_md1[:, :, 0], cmap='gray')
    plt.title('Cov Layer 1')

    plt.subplot(222)
    plt.imshow(out_md2[:, :, 0], cmap='gray')
    plt.title('Maxpooling Layer 1')

    plt.subplot(223)
    plt.imshow(out_md3[:, :, 0], cmap='gray')
    plt.title('Cov Layer 2')

    plt.subplot(224)
    plt.imshow(out_md4[:, :, 0], cmap='gray')
    plt.title('Maxpooling Layer 2')
    plt.show()