"""
mnist_loader from Michael Nielsen's 
Code samples for "Neural Networks and Deep Learning"
~~~~~~~~~~~~

"""

#### Libraries

# Third-party libraries
import numpy as np
from sklearn import preprocessing

# Code has been modified from original to process csv data directly

def load_data():

    trf = '../data/training_data.csv'
    vf = '../data/validation_data.csv'
    tef = '../data/test_data.csv'

    training_data = np.genfromtxt(trf, delimiter=',', dtype='float',filling_values=0, skip_header=1)

    validation_data = np.genfromtxt(vf, delimiter=',', dtype='float',filling_values=0, skip_header=1)

    test_data = np.genfromtxt(tef, delimiter=',', dtype='float',filling_values=0, skip_header=1)

    return (training_data, validation_data, test_data)
    #return (training_data, test_data)

def load_data_wrapper():

    tr_d, va_d, te_d = load_data()
    #tr_d, te_d = load_data()

    ## ti1 points to all the input data, starting from col B
    ti1 = tuple(x[1:] for x in tr_d)
    #norm_ti1 = preprocessing.normalize(ti1, norm='l2')
    norm_ti1 = preprocessing.MinMaxScaler().fit_transform(ti1)
    training_inputs = [np.reshape(x, (-1, 1)) for x in norm_ti1]

    #for row in training_inputs:
       #print (row)

    ## ti2 points to all the letter grade results data from col A
    ti2 = tuple(x[0] for x in tr_d)
    training_results = [vectorized_result(y) for y in ti2]

    #print ("\n")
    #for row in training_results:
       #print (row)

    training_data = zip(training_inputs, training_results)

    #print ("\n\n")
    #print (training_data)

    ## vi1 points to all the input data, starting from col B
    vi1 = tuple(x[1:] for x in va_d)
    ## vi2 points to all the letter grade results data from col A
    vi2 = tuple(x[0] for x in va_d)
    validation_inputs = [np.reshape(x, (-1, 1)) for x in vi1]

    validation_data = zip(validation_inputs, vi2)

    ## te1 points to all the input data, starting from col B
    te1 = tuple(x[1:] for x in te_d)
    #norm_te1 = preprocessing.normalize(te1, norm='l2')
    norm_te1 = preprocessing.MinMaxScaler().fit_transform(te1)
    ## te2 points to all the letter grade results data from col A
    te2 = tuple(x[0] for x in te_d)
    test_inputs = [np.reshape(x, (-1, 1)) for x in norm_te1]

    test_data = zip(test_inputs, te2)

    return (training_data, validation_data, test_data)
    #return (training_data, test_data)


def vectorized_result(j):
    """Return a 5-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert the grade digit
    (1..5) into a corresponding desired output from the neural
    network."""

    e = np.zeros((5, 1))
    e[int(j) - 1] = 1.0

    return e


# main method used for testing the methods above
'''def main():
    load_data_wrapper()

main ()'''
