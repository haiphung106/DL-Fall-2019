"""
Created by haiphung106
"""
import tarfile
import pickle
import random
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm

cifar10_dataset_path = 'cifar-10-python'


def load_cifar10_batch(cifar10_dataset_path, batch_id):
    with open(cifar10_dataset_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].shape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def onehot(x, labels):
    y = np.zeros((len(x), labels))
    for index, value in enumerate(x):
        y[index][value] = 1
    return y


def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)
    pickle.dump((features, labels), open(filename, 'wb'))


def final_pre_and_save_data(cifar10_dataset_path, normalize, onehot):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_id in range(1, n_batches + 1):
        features, labels = load_cifar10_batch(cifar10_dataset_path, batch_id)

        # find index for the validation data in the whole data set(10%)
        idx_of_validation = int(len(features) * 0.1)
        # pre-process the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the labels
        # - save in a new file named, "preprocess_batch" + batch_number
        # - each file for each batch
        preprocess_and_save(normalize, onehot, features[:-idx_of_validation], labels[:-idx_of_validation],
                            'preprocess_batch' + str(batch_id) + '.p')

        valid_features.extend(features[-idx_of_validation])
        valid_labels.extend(labels[-idx_of_validation:])

    preprocess_and_save(normalize, onehot, np.array(valid_features), np.array(valid_labels), 'preprocess_validation.p')

    # Load and pre-process data
    with open(cifar10_dataset_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

        test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = batch['labels']

    # Save pre-process
    preprocess_and_save(normalize, onehot, np.array(test_features), np.array(test_labels), 'preprocess_training.p')
