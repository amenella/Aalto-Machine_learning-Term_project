# coding=utf-8

from sklearn import metrics
import numpy as np

def delete_id_column(x):
    x = np.delete(x, 0, 1)
    return x

def insert_first_column(array, value):
    array = np.insert(array, 0, value, axis=1)
    return array

def extract_xy(data):

    col_nb = len(data[0][:])
    # line_nb = len(data[:])

    x = np.delete(data, col_nb - 1, 1)

    x = delete_id_column(x)

    #  insert 1 column of 1 at the beginning of x
    x = insert_first_column(x, 1)

    y = list()
    for a in data:
        b = a[col_nb - 1]
        y.append(b)

    return (x, y)


def compute_regression_coeff(data):

    x, y = extract_xy(data)

    xT =  np.transpose(x)
    xT_x = np.dot(xT, x)
    xT_x_inversed = np.linalg.inv(xT_x)

    beta = np.dot(xT_x_inversed, xT)
    beta = np.dot(beta, y)

    return beta


def linear_regression(beta, x):

    output_data = list()

    for elem in x:
        computed_y = np.dot(elem, beta)
        output_data.append(computed_y)

    return output_data


def test_regression(training_data, testing_data, testing_data_solutions):

    print "Test regression"


    # delete id column
    testing_data = delete_id_column(testing_data)
    testing_data_solutions = delete_id_column(testing_data_solutions)

    #  insert 1 column of 1 at the beginning of the testing_data array
    testing_data = insert_first_column(testing_data, 1)

    # compute regression coefficients for the training data
    beta = compute_regression_coeff(training_data)

    # do linear regression :
    # with the computed regression coefficients from the training data
    # on the testing data
    computed_testing_data = linear_regression(beta, testing_data)

    print computed_testing_data

    mse = metrics.mean_squared_error(testing_data_solutions, computed_testing_data)
    print "Mean squared error :", mse

    return "Done"
