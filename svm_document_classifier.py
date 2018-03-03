import random

import numpy as np

from multinomial_naive_bayes import error, get_indicative_tokens


def read_matrix(file):
    fd = open(file, 'r')
    header = fd.readline()
    n_row, n_col = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((n_row, n_col))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    fd.close()
    Y = [1 if item == 1 else -1 for item in Y]
    return matrix, tokens, np.array(Y)


def preprocess_data(data_matrix):
    data_matrix = 1. * (data_matrix > 0)
    return data_matrix


def gauss_kernel_matrix(data_matrix, tau=8):
    norms = np.sum(data_matrix**2, axis=-1)
    gram = data_matrix.dot(data_matrix.T)
    K = np.exp((-1/(tau**2))*(-2*gram + norms.reshape((1, -1)) + norms.reshape((-1, 1))))
    return K


def train(data_matrix, labels, reg=1/64, outer_loops=40, tau=8):
    kernel_matrix = gauss_kernel_matrix(data_matrix)
    m = kernel_matrix.shape[0]
    alpha = np.zeros(m)
    lr = 1
    alpha_avg = np.zeros(m)
    for i in range(outer_loops):
        seq = random.sample(range(m), m)
        for s, j in enumerate(seq):
            lr = 1 / np.sqrt((i+1)*(s+1))
            alpha -= lr * reg * kernel_matrix[j]*alpha[j]
            if labels[j]*(kernel_matrix[j].dot(alpha)) < 1:
                alpha += lr * labels[j]*kernel_matrix[j]
            alpha_avg += alpha
    alpha_avg /= outer_loops*m
    parameters = dict()
    parameters['alpha'] = alpha_avg
    parameters['train_matrix'] = data_matrix
    parameters['tau'] = tau
    return parameters


def predict(parameters, test_matrix):
    tau = parameters['tau']
    alpha = parameters['alpha']
    train_matrix = parameters['train_matrix']
    train_norms = np.sum(train_matrix**2, axis=-1)
    test_norms = np.sum(test_matrix**2, axis=-1)
    gram = train_matrix.dot(test_matrix.T)
    K = np.exp((-1/(tau**2))*(train_norms.reshape(-1, 1) + test_norms.reshape(1, -1) - 2*gram))
    preds = np.sign(alpha.dot(K))
    return preds


def main():
    train_files = ('MATRIX.TRAIN.50', 'MATRIX.TRAIN.100', 'MATRIX.TRAIN.200',
                   'MATRIX.TRAIN.400', 'MATRIX.TRAIN.800', 'MATRIX.TRAIN.1400', 'MATRIX.TRAIN')
    test_matrix, tokens, test_labels = read_matrix('MATRIX.TEST')
    test_matrix = preprocess_data(test_matrix)
    for file in train_files:
        train_matrix, tokens, train_labels = read_matrix(file)
        train_matrix = preprocess_data(train_matrix)
        params = train(train_matrix, train_labels)

        predictions = predict(params, test_matrix)
        print(file + ' : Error: %1.4f' % error(test_labels, predictions))


if __name__ == "__main__":
    main()
