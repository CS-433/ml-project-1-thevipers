# -*- coding: utf-8 -*-
"""cross validation"""
import numpy as np
from implementations import *
from proj1_helpers import *
from preprocessing import *

def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the split ratio.
    """
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, y_tr, x_te, y_te

def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


#Commenter +++++
#En gros c'est vraiment la fonction de base de cross validation, masi ça marche avec n'importe laquelle des 6 methodes
#A revoir avec gamma tunable
def cross_validation(y, x, k_indices, k_fold, method, degree=1, lambda_=None, log=False, **kwargs):
    """
    Return the training and test accuracies obtained by cross validation for all the 6 different methods
    """
    
    acc_train=[]
    acc_test=[]
    
    #Cross-validation repeated for each fold (at each iteration the test set is different)
    for k in range(k_fold) :
        
        #Split the train and test sets
        test_i = k_indices[k]
        train_i = (np.delete(k_indices, k, axis=0)).reshape(-1)
        test_x = x[test_i]
        test_y = y[test_i]
        train_x = x[train_i]
        train_y = y[train_i]

        # form data with polynomial degree:
        train_x = build_poly(train_x, degree)
        test_x = build_poly(test_x, degree)
        
        #Compute the weights
        if(lambda_==None) :
            w, loss = method(train_y, train_x, **kwargs)
        else :
            w, loss = method(train_y, train_x, lambda_, **kwargs)

        # calculate the accuracy for train and test data
        acc_train.append(accuracy(train_y, train_x, w, log))
        acc_test.append(accuracy(test_y, test_x, w, log))
    
    #Average the accuracies over the 'k_fold' folds
    acc_tr = np.mean(acc_train)
    acc_te = np.mean(acc_test)
    
    return acc_tr, acc_te
    
    
#En gros ça ça fait la même chose que les fonctions de Camille qui redonnent le meilleur degree, 
#mais ça marche pour toutes les methodes
#MARCHE PAS POUR L'INSTANT CAR GAMMA PAS TUNABLE, A REVOIR QUAND JE PEUX
def cross_validation_best_degree(y, x, k_indices, k_fold, method, degrees, lambda_=None, log=False, **kwargs) : 
    """
    commenter ++++++
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best gammas and the associated test and train error
    best_gammas = []
    best_err_te = []
    best_err_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the loss of training data and test data for each degree
        err_tr = []
        err_te = []
        # cross validation to find the best learning rate gamma for each degree
        for gamma in gammas:
            #Call the function cross validation which returns the mean accuracy of the training and the test set
            acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, degree, lambda_, log, **kwargs)
            # store the mean error over the k-folds for each lambda
            err_tr.append(acc_tr)
            err_te.append(acc_te)     
        # find the optimal lambda which lead to the minimum test error for the current degree
        # and store the optimal lambda, the minimum test error, and the train error for the same lambda 
        ind_gamma_opt = np.argmin(err_te)
        best_gammas.append(gammas[ind_gamma_opt])
        best_err_te.append(err_te[ind_gamma_opt])
        best_err_tr.append(err_tr[ind_gamma_opt])
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(best_err_te)      
        
    return degrees[ind_best_degree], best_err_te[ind_best_degree], best_err_tr[ind_best_degree]