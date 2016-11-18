# coding=utf-8

# Importing modules:

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import regression as rg
import classification as cl
import os
from sklearn.naive_bayes import MultinomialNB

dir = os.path.dirname(__file__)

def load_file(c):
    return np.loadtxt(c,dtype=int,delimiter=',',skiprows=1)


if __name__ == '__main__':
    training_file = os.path.join(dir,'data/classification_dataset_training.csv')
    test_file = os.path.join(dir,'data/classification_dataset_testing.csv')
    solution_file = os.path.join(dir,'data/classification_dataset_testing_solution.csv')
    training=load_file(training_file)#matrice with 5000 rows and 52 columns
    test2D=load_file(test_file)
    solution=load_file(solution_file)

    likelihood0, likelihood1, prior0, prior1=cl.apprentissage_multinomiale(training)


    prediction=cl.test(likelihood0,likelihood1,prior0,prior1,test2D)
    sol=cl.extract(solution)
    pourcentage=0.
    for i in range(0,1000):
       if(sol[i]==prediction[i]):
            pourcentage+=1
    pourcentage=float(pourcentage)/1000.
    print(pourcentage)
