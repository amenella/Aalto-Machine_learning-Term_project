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

def load_file(c):
    return np.loadtxt(c,dtype=int,delimiter=',',skiprows=1)


if __name__ == '__main__':
    filename = os.path.join(dir,'data/classification_dataset_training.csv')
    matrice=load_file(filename)#matrice with 5000 rows and 52 columns
    print(matrice[0][51])
    print rg.test_regression()

    print cl.test_classification()
