# coding=utf-8

# Importing modules:

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import regression as rg
import classification as cl
import os

dir = os.path.dirname(__file__)

def load_file(filename):
    return np.loadtxt(filename,dtype=int,delimiter=',',skiprows=1)


if __name__ == '__main__':

    classification_training_file = os.path.join(dir,'data/classification_dataset_training.csv')

    classification_training = load_file(classification_training_file) # matrice with 5000 rows and 52 columns



    regression_training_file = os.path.join(dir,'data/regression_dataset_training.csv')
    regression_testing_file = os.path.join(dir,'data/regression_dataset_testing.csv')
    regression_testing_solution_file = os.path.join(dir,'data/regression_dataset_testing_solution.csv')

    regression_training = load_file(regression_training_file) # matrice with 5000 rows and 52 columns
    regression_testing = load_file(regression_testing_file) # matrice with 5000 rows and 52 columns
    regression_testing_solution = load_file(regression_testing_solution_file) # matrice with 5000 rows and 52 columns


    print rg.test_regression(regression_training, regression_testing, regression_testing_solution)

    print cl.test_classification()
