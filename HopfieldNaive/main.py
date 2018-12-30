"""First approach

http://neupy.com/2015/09/20/discrete_hopfield_network.html
https://www.doc.ic.ac.uk/~ae/papers/Hopfield-networks-15.pdf
"""
import itertools
from typing import List

import numpy as np


size = [6, 7]   # row, column
a_letter = [0, 0, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 1, 1, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0]

b_letter = [0, 1, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 1, 1, 0, 0]

d_letter = [0, 1, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 1, 1, 0, 0]

c_letter = [0, 0, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 0, 0]

o_letter = [0, 0, 1, 1, 0, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 0, 0]

one_sign = [0, 0, 0, 1, 0, 0,
            0, 0, 1, 1, 0, 0,
            0, 1, 0, 1, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 1, 1, 1, 1, 1]

zero_sign = [0, 0, 1, 1, 0, 0,
             0, 1, 0, 0, 1, 0,
             0, 1, 0, 0, 1, 0,
             0, 1, 0, 0, 1, 0,
             0, 1, 0, 0, 1, 0,
             0, 1, 0, 0, 1, 0,
             0, 0, 1, 1, 0, 0]

test_sing = [0, 0, 0, 1, 0, 0,
             0, 0, 0, 1, 0, 0,
             0, 0, 0, 1, 0, 0,
             0, 0, 0, 1, 0, 0,
             0, 0, 0, 1, 0, 0,
             0, 0, 0, 1, 0, 0,
             0, 1, 1, 1, 1, 1]


def normalise(data_to_normalise: List):
    return np.array([elmn*2 - 1 for elmn in data_to_normalise])


def sign(matrix: np.matrix):
    matrix[matrix >= 0] = 1.0
    matrix[matrix < 0] = -1.0
    return matrix


def main(sync_mode=True):
    training_examples = [d_letter, c_letter, one_sign]
    to_recognise = np.array([normalise(one_sign)])

    training_examples_normalised = [normalise(dat) for dat in training_examples]
    training_matrix = np.matrix(training_examples_normalised)

    weights_matrix = training_matrix.T * training_matrix
    np.fill_diagonal(weights_matrix, 0)

    features_number = weights_matrix.shape[0]
    max_memory = np.ceil(features_number / (2 * np.log(features_number)) - 1)
    print('Shape features = {}'.format(features_number))
    print('Max elements in memory = {}'.format(max_memory))

    if sync_mode:
        predicted = sign(to_recognise.dot(weights_matrix))
    else:
        predicted = to_recognise
        for _ in range(500):
            rnd_position = np.random.random_integers(0, features_number - 1)
            predicted[0, rnd_position] = predicted.dot(weights_matrix[:, rnd_position])[0][0]
            predicted = sign(predicted)
    predicted = list(itertools.chain.from_iterable(np.matrix.tolist(predicted)))
    for y_axis in range(size[1]):
        print(str(predicted[y_axis * size[0]: size[0] * (y_axis + 1) - 1])
              .strip(']').strip('[').replace(',', '').replace(' ', '').replace('-1.0', ' ').replace('1.0', '*')
              .replace('-1', ' ').replace('1', '*'))


if __name__ == '__main__':
    main()
