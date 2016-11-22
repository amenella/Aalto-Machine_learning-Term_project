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
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
# from sklearn.neural_network import MLPClassifier


dir = os.path.dirname(__file__)

def load_file(filename):
    return np.loadtxt(filename,dtype=int,delimiter=',',skiprows=1)


def plot_errors(figure_fname, x_label, y_label, mse_value, nb_features):




    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(nb_features, mse_value, c='r', alpha=0.5)
    # ax.scatter(average_degrees, second_largest_components, c='g', alpha=0.5, label="Second largest CC")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # ax.legend(loc=2)

    fig.savefig(figure_fname, ftype='pdf')



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



    figure_fname = 'Mean square errors in function of nb of selected features'
    x_label = 'Nb of features'
    y_label = 'MSE'

    fig = plt.figure()
    ax = fig.add_subplot(111)


    # ax.scatter(average_degrees, second_largest_components, c='g', alpha=0.5, label="Second largest CC")



    all_mse=list()
    all_min_mse=list()
    all_k_best=list()
    all_min_k_best=list()

    all_mser = list()

    for K_Best in range(7, 51):
        freg = SelectKBest(f_regression,k=K_Best)
        X_train = freg.fit_transform(features, target)
        a = freg.get_support(indices=True)
        test2Dfs = test_feature[0:1000, a]
        #valeur d'adrien avant:0.0475212809951
        #LINEAR REGRESSION
        mse=rg.test_regression(X_train,test2Dfs, regression_testing_solution,target)

        all_mse.append(mse)
        all_k_best.append(K_Best)

        if mse_min > mse:
            all_min_mse.append(mse)
            all_min_k_best.append(K_Best)
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

            all_mser.append(mser)
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


    ax.plot(all_k_best, all_mse, c='r', label="linear regression")
    # ax.plot(alpha_ridge, all_mser, c='b', label="ridge regression")
    # ax.scatter(all_min_k_best, all_min_mse, c='r')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # ax.legend(loc=2)

    fig.savefig(figure_fname, ftype='pdf')

    plt.show()


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