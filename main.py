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

def load_file():
    return "Load file"


if __name__ == '__main__':

    print load_file()

    print rg.test_regression()

    print cl.test_classification()
