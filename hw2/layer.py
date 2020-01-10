"""
Created by haiphung106
"""
import numpy as np
import pickle
def Convolution(input, stride, activation, kernel, bias):
    """
    :param input: m*m*d
    :param stride: matrix dimension s*s
    :param activation: relu
    :param kernel: 3D tensor, with dimension r*r*d*k
    :param bias: 2D vector with dimension k*1
    :return: output = 3D tensor with dimension m*m*k
    """
    d = input.shape[2]
    m = input.shape[0]
    s = stride[0]
    r = kernel.shape[0]
    k = kernel.shape[3]
    p = (m*s - m + s + r)/2
    p = int(p)
    k = kernel.shape[3]
    output = np.zeros((m, m, k))
    for i in range(k):
        slice_output = np.zeros((m, m))
        for j in range(d):
            slice_input = input[:, :, j]
            slice_kernel = kernel[:, :, j, i]

            padding_input = np.zeros((m + 2*p, m + 2*p))
            padding_input[p:m+p, p:m+p] = slice_input
            temp_output = np.zeros((m, m))
            for row in range(m):
                for col in range(m):
                    temp = padding_input[row*s:row*s + r, col*s:col*s + r]*slice_kernel
                    temp_output[row, col] = np.sum(temp)
            slice_output = slice_output + bias[i]
            if(activation == 'relu'):
                slice_output = np.clip(slice_output, 0, None)
                output[:, :, i] = slice_output
    return output

def MaxPooling(input, pool_size):
    """
    :param input: I = 3D tensor with dimension m*m*d
    :param pool_size: 2D matrix with dimension pl*pl
    :return: 3D tensor with dimension n*n*d = (m/pl)*(m/pl)*d
    """
    d = input.shape[2]
    m = input.shape[0]
    pl = pool_size[0]
    n = int(m/pl)
    output = np.zeros((n, n, d))
    for i in range(d):
        slice_input = input[:, :, i]
        temp = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                temp[j, k] = np.max(slice_input[j * pl:(j+1)*pl, k*pl:(k+1)*pl])
        output[:, :, i] = temp
    return output

def Fllatening(input):
    '''
    :param input: a*a*k
    :return: (a*a*k, 1)
    '''
    output =np.reshape(input, (-1, 1))
    return output

def FullyConnected(input, weight, bias, activation):
    bias = np.reshape(bias, (-1, 1))
    output = np.dot(weight.T, input) + bias
    if activation == 'relu':
        output = np.clip(output, 0, None)
    elif activation == 'softmax':
        output = np.exp(output)
        sum = np.sum(output)
        output = output / sum
    return output
strides = (1, 1)
pool_size = (2, 2)
kernel_size = (3, 3)

def run_forward(input_point, model_weight):
    output = Convolution(input=input_point, stride= strides, activation='relu', kernel=model_weight[0], bias=model_weight[1])
    output = MaxPooling(input = output, pool_size = pool_size)
    output = Convolution(input=input_point, stride= strides, activation='relu', kernel=model_weight[2], bias=model_weight[3])
    output = MaxPooling(input=output, pool_size=pool_size)
    output = Fllatening(input = output)
    output = FullyConnected(input = output, weight= model_weight[4], bias=model_weight[5], activation='relu')
    output = FullyConnected(input=output, weight=model_weight[4], bias=model_weight[7], activation='relu')
    output = np.reshape(output, (-1, 1))
    return output

def check_run_forward(input_data, model, model_weight):
    check_result = False
    pick_random_index = np.random.choice(input_data.shape[0], 100)
    data_for_check = input_data[pick_random_index]
    for i in range(data_for_check.shape[0]):
        my_fw = run_forward(input_data[i], model_weight)
        nn_fw = model.predict(input_data[i].reshape(1, 28, 28, 1))
        if ((np.argmax(my_fw) == np.argmax(nn_fw))):
            check_result = True
        else:
            check_result = False
        print('data_index {}, check_flag {}'.format(pick_random_index[i], check_result))
    return check_result

cifar10_dataset_path = 'cifar-10-python'


def load_cifar10_batch(cifar10_dataset_path):
    with open(cifar10_dataset_path, mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data']
    labels = batch['labels']
    return features, labels