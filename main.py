# coding=utf-8

# Importing modules:

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import csv
import regression as rg
import classification as cl
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neural_network import MLPClassifier


dir = os.path.dirname(__file__)

def load_file(filename):
    return np.loadtxt(filename,dtype=int,delimiter=',',skiprows=1)


if __name__ == '__main__':

    ##############################################################################
    # REGRESSION

    regression_training_file = os.path.join(dir,'data/regression_dataset_training.csv')
    regression_testing_file = os.path.join(dir,'data/regression_dataset_testing.csv')
    regression_testing_solution_file = os.path.join(dir,'data/regression_dataset_testing_solution.csv')

    regression_training = load_file(regression_training_file) # matrice with 5000 rows and 52 columns
    regression_testing = load_file(regression_testing_file) # matrice with 5000 rows and 52 columns
    regression_testing_solution = load_file(regression_testing_solution_file)  # matrice with 5000 rows and 52 columns
    sol = np.delete(regression_testing_solution,0,1)
    test_feature = regression_testing[0:1000, 1:51]
    features = regression_training[0:5000, 1:51]
    #features = StandardScaler().fit_transform(features)
    target = regression_training[0:5000, 51]
    sol = regression_testing_solution[0:1000,1]

    # FEATURE SELECTION
    best_number_features_ridge_regression=0
    best_number_features_linear_regression=0
    mse_min=10
    mse_ridge_min=10

    for K_Best in range(2, 51):
        freg = SelectKBest(f_regression,k=K_Best)
        X_train = freg.fit_transform(features, target)
        a = freg.get_support(indices=True)
        test2Dfs = test_feature[0:1000, a]
        #valeur d'adrien avant:0.0475212809951
        #LINEAR REGRESSION
        mse=rg.test_regression(X_train,test2Dfs, regression_testing_solution,target)

        if mse_min > mse:
            best_number_features_linear_regression= K_Best
            mse_min = mse
        #RIDGE REGRESSION
        #we test for different alpha which represent complexity penalizer to avoid everfitting(draw a graph in function of alpha for the best number of features)
        alpha_ridge = [0.003,0.0025,0.0024,0.0023,0.0022,0.0021,0.002,0.001,0.0001,0.00001]
        for i  in alpha_ridge:
            clf = Ridge(alpha=i,normalize=True)
            clf.fit(X_train,target)
            y_pred = clf.predict(test2Dfs)
            mser = metrics.mean_squared_error(sol,y_pred)
            if mse_ridge_min > mser:
                best_number_features_ridge_regression = K_Best
                mse_ridge_min = mser
                alpha=i


    # freg = SelectKBest(f_regression,k=11)
    # X_train = freg.fit_transform(features, target)
    # a = freg.get_support(indices=True)
    # test2Dfs = test_feature[0:1000, a]
    # clf = Ridge(alpha=i, normalize=True)
    # clf.fit(X_train, target)
    # y_pred = clf.predict(test2Dfs)




    # freg = SelectKBest(f_regression, k=10)
    # X_train = freg.fit_transform(features, target)
    # a = freg.get_support(indices=True)
    # test2Dfs = test_feature[0:1000, a]
    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,random_state = 1)
    # clf.fit(X_train,target)
    # y_neural_pred=clf.predict(test2Dfs)
    # mseneural = metrics.mean_squared_error(sol, y_neural_pred)
    # print('mseneural'+str(mseneural))
    # nn = Classifier(
    #     layers=[
    #         Layer("Maxout", units=100,pieces=2),
    #         Layer("Softmax")],
    #     learning_rate=0.001,
    #     n_iter=25)
    # nn.fit(X_train,target)
    # y_example = nn.predict(test2Dfs)
    # mseneural = metrics.mean_squared_error(sol, y_example)
    # print('mseneural' + str(mseneural))
    print("Pr la regression lineaire la mse est:" + str(mse_min))
    print("le nombre de feature est :" + str(best_number_features_linear_regression))
    print("Pour la ridge regression la mse est:" + str(mse_ridge_min)+"avec alpha="+str(alpha))
    print("le nombre de feature est :" + str(best_number_features_ridge_regression))
    ###############################################################################
    # CLASSIFICATION
    #path of file
    training_file = os.path.join(dir,'data/classification_dataset_training.csv')
    test_file = os.path.join(dir,'data/classification_dataset_testing.csv')
    solution_file = os.path.join(dir,'data/classification_dataset_testing_solution.csv')
    #creation of matrix associated to data
    training=load_file(training_file)#matrice with 5000 rows and 52 columns
    test2D=load_file(test_file)
    solution=load_file(solution_file)

    features = training[0:5000, 1:51]
    #features=StandardScaler().fit_transform(features)
    target = training[0:5000,51]
    test_feature = test2D[0:1000,1:51]
    #feature selection of the feature training data
    best_number_features=0
    pourcentage_max=0
    for K_Best in range(2,51):
        ch2 = SelectKBest(chi2, k=K_Best)
        X_train = ch2.fit_transform(features, target)
        a=ch2.get_support(indices=True)
        test2Dfs = test_feature[0:1000,a]
        #multinomial naive bayes learning algorithm
        likelihood0, likelihood1, prior0, prior1=cl.apprentissage_multinomiale(X_train,target)
        prediction=cl.test(likelihood0,likelihood1,prior0,prior1,test2Dfs)
        sol=cl.extract(solution)
        pourcentage=0.
        for i in range(0,1000):
            if(sol[i]==prediction[i]):
                pourcentage+=1
        pourcentage=float(pourcentage)/1000.
        if pourcentage_max<pourcentage:
            best_number_features=K_Best
            pourcentage_max=pourcentage
        #print(pourcentage)
    print("Classification:")
    print("la reussite est de:"+str(pourcentage_max))
    print("le nombre de feature est :"+str(best_number_features))


    #FOR KAGGLE PROJECT
    # ch2 = SelectKBest(chi2, k=21)
    # X_train = ch2.fit_transform(features, target)
    # a = ch2.get_support(indices=True)
    # test2Dfs = test_feature[0:1000, a]
    # # multinomial naive bayes learning algorithm
    # likelihood0, likelihood1, prior0, prior1 = cl.apprentissage_multinomiale(X_train, target)
    # prediction = cl.test(likelihood0, likelihood1, prior0, prior1, test2Dfs)
    # sol = cl.extract(solution)
    # fname = "out.csv"
    # file = open(fname, "wb")
    #
    #
    # try:
    #     #
    #     # Création de l'''écrivain'' CSV.
    #     #
    #     writer = csv.writer(file)
    #
    #     #
    #     # Écriture de la ligne d'en-tête avec le titre
    #     # des colonnes.
    #     writer.writerow(('ID', 'Rating'))
    #     for i in range(0, 1000):
    #         if(prediction[i]==1.0):
    #             writer.writerow((solution[i][0],0.9))
    #         else:
    #             writer.writerow((solution[i][0], 0.1))
    # finally:
    #     #
    #     # Fermeture du fichier source
    #     #
    #     file.close()