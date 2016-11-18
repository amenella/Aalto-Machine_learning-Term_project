# coding=utf-8
import numpy as np
import math

def apprentissage_multinomiale(tableau2D):
    N0=0#number of review in classe 0
    N1=0#number of reviex in classe 1
    tf0=np.zeros(50)#array of class 0 where case i is the frequency of word i ib ckass 0
    tf1=np.zeros(50)
    D0=0#number of frequency of all word for class0
    D1=0
    likelihood0=np.zeros(50)
    likelihood1=np.zeros(50)
    for i in range(0,5000):
        k=tableau2D[i,51]#current class
        if (tableau2D[i,51]==1):
            N1+=1
        else:
            N0+=1
        for j in range(1,51):
            if (k==0):
                tf0[j-1]+=tableau2D[i][j]
                D0+=tf0[j-1]
            else:
                tf1[j-1]+= tableau2D[i][j]
    D0=sum(tf0)
    D1=sum(tf1)
    prior0=float(N0)/5000.#prior probability of class 0
    prior1=float(N1)/5000.
    for i in range(0,50):
        likelihood0[i]=float((tf0[i]+1)/(D0+50))#likelihood of each term of the vocabulary given its class
        likelihood1[i]=float((tf1[i]+1)/(D1+50))
    return (likelihood0,likelihood1,prior0,prior1)

#analyze the solution file for test data and return the class corresponding at each document in a array
def extract(tableau2D):
    solution=np.zeros(1000)
    for i in range(0,1000):
        solution[i]=tableau2D[i][1]
    return solution

def test(likelihood0,likelihood1,prior0,prior1,test2D):
    prediction=np.zeros(1000)#array to save the value of the predict class corresponding to teach documents
    for i in range(0,1000):
        P0 = math.log(prior0)
        P1 = math.log(prior1)
        for j in range(1,51):
            P0+=test2D[i][j]*math.log(likelihood0[j-1])
            P1+=test2D[i][j]*math.log(likelihood1[j-1])
        if(P0>P1):#we attribute the class which posterior probability is greater than the others
            prediction[i]=0
        else:
            prediction[i]=1
    return prediction





