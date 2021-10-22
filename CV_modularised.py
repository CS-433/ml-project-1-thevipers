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



def cross_validation(y, x, k_indices, k_fold, method, degree=1, lambda_=None, gamma=None, log=False, **kwargs):
    """
    Return the training and test accuracies obtained by cross validation for any of the 6 different methods
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
            if(gamma==None) :
                w, loss = method(train_y, train_x, **kwargs)
            else :
                w, loss = method(train_y, train_x, gamma=gamma, **kwargs)
        else :
            if(gamma==None) :
                w, loss = method(train_y, train_x, lambda_=lambda_, **kwargs)
            else :
                w, loss = method(train_y, train_x, lambda_=lambda_, gamma=gamma, **kwargs)

        # calculate the accuracy for train and test data
        acc_train.append(accuracy(train_y, train_x, w, log))
        acc_test.append(accuracy(test_y, test_x, w, log))
    
    #Average the accuracies over the 'k_fold' folds
    acc_tr = np.mean(acc_train)
    acc_te = np.mean(acc_test)
    
    return acc_tr, acc_te


def tune_best_one(y, x, k_fold, method, seed, params, name='degree', log=False, **kwargs) :
    """
    Tune one of the following parameters : degree, lambda or gamma
    This function can take any of the six methods and tune the parameter we want to optimize.
    It returns the optimal parameter, the best test error and the best training error
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best gammas and the associated test and train error
    best_err_te = []
    best_err_tr = []
    # vary degree
    for param in params:
        #Call the function cross validation which returns the mean accuracy of the training and the test set
        if(name== 'degree') :
            acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, degree=param, log=log, **kwargs)
        elif(name== 'lambda') :
            acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, lambda_=param, log=log, **kwargs)
        elif(name== 'gamma'):
            acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, gamma=param, log=log, **kwargs)
        else :
            raise NameError(name, ' is not a name of one of the tunable parameters')
        # store the mean error over the k-folds for each degree
        best_err_tr.append(acc_tr)
        best_err_te.append(acc_te)     
    # find the degree which leads to the minimum test error
    ind_best_param =  np.argmin(1-best_err_te)      
        
    return params[ind_best_param], 1-best_err_te[ind_best_param], 1-best_err_tr[ind_best_param]
    
    

####################################################################################################################
#The following functions aim to tune parameters simultaneously. It can be better in general, but are a bit heavy in term of 
#computational cost.

def tune_best_deg_lam_gam(y, x, k_fold, method, degrees, lambdas, gammas, log=False, seed=1, **kwargs) : 
    """
    commenter ++++++
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best gammas and the associated test and train error
    best_lambdas = []
    best_gammas = []
    best_err_te = []
    best_err_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the loss of training data and test data for each degree
        err_tr = []
        err_te = []
        best_gammas_tmp = []
        #vary lambda
        for lambda_ in lambdas:
            # define lists to store the loss of training data and test data for each degree
            err_tr_gam = []
            err_te_gam = []
            #vary lambda
            for gamma in gammas:
                #Call the function cross validation which returns the mean accuracy of the training and the test set
                acc_tr, acc_te = cross_validation(y, x, k_indices=k_indices, k_fold=k_fold, method=method, degree=degree,
                                                                        lambda_=lambda_, gamma=gamma, log=log, **kwargs)
                # store the mean error over the k-folds for each lambda
                err_tr_gam.append(1-acc_tr)
                err_te_gam.append(1-acc_te)  
                
            # find the optimal gamma which lead to the minimum test error for the current degree and lambda
            # and store the optimal gamma, the minimum test error, and the train error for each lambdas
            ind_gam_opt = np.argmin(err_te_gam)
            best_gammas_tmp.append(gammas[ind_gam_opt])
            err_te.append(err_te_gam[ind_gam_opt])
            err_tr.append(err_tr_gam[ind_gam_opt])
            
        # find the optimal lambda which lead to the minimum test error for the current degree
        # and store the optimal lambda, the minimum test error, and the train error for the same lambda 
        ind_lambda_opt = np.argmin(err_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_gammas.append(best_gammas_tmp[ind_lambda_opt])
        best_err_te.append(err_te[ind_lambda_opt])
        best_err_tr.append(err_tr[ind_lambda_opt])
            
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(best_err_te)
    print(best_err_te)
        
    return degrees[ind_best_degree], best_gammas[ind_best_degree], best_lambdas[ind_best_degree], best_err_te[ind_best_degree], best_err_tr[ind_best_degree]

    
    

def tune_best_deg_gam(y, x, k_fold, method, degrees, gammas, log=False, seed=1, **kwargs) : 
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
        # vary gamma
        for gamma in gammas:
            #Call the function cross validation which returns the mean accuracy of the training and the test set
            acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, degree=degree, gamma=gamma, log=log, **kwargs)
            # store the mean error over the k-folds for each lambda
            err_tr.append(1-acc_tr)
            err_te.append(1-acc_te)     
        # find the optimal lambda which lead to the minimum test error for the current degree
        # and store the optimal lambda, the minimum test error, and the train error for the same lambda 
        ind_gamma_opt = np.argmin(err_te)
        best_gammas.append(gammas[ind_gamma_opt])
        best_err_te.append(err_te[ind_gamma_opt])
        best_err_tr.append(err_tr[ind_gamma_opt])
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(best_err_te)

    return degrees[ind_best_degree], best_gammas[ind_best_degree], best_err_te[ind_best_degree], best_err_tr[ind_best_degree]



def tune_best_deg_lam(y, x, k_fold, method, degrees, lambdas, log=False, seed=1, **kwargs) : 
    """
    commenter ++++++
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best gammas and the associated test and train error
    best_lambdas = []
    best_err_te = []
    best_err_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the loss of training data and test data for each degree
        err_tr = []
        err_te = []
        #vary lambda
        for lambda_ in lambdas:
            #Call the function cross validation which returns the mean accuracy of the training and the test set
            acc_tr, acc_te = cross_validation(y, x, k_indices=k_indices, k_fold=k_fold, method=method, degree=degree,
                                                                                      lambda_=lambda_, log=log, **kwargs)
            # store the mean error over the k-folds for each lambda
            err_tr.append(1-acc_tr)
            err_te.append(1-acc_te)     
        # find the optimal lambda which lead to the minimum test error for the current degree
        # and store the optimal lambda, the minimum test error, and the train error for the same lambda 
        ind_lambda_opt = np.argmin(err_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_err_te.append(err_te[ind_lambda_opt])
        best_err_tr.append(err_tr[ind_lambda_opt])
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(best_err_te)      
        
    return degrees[ind_best_degree], best_lambdas[ind_best_degree], best_err_te[ind_best_degree], best_err_tr[ind_best_degree]



def tune_best_deg(y, x, k_fold, method, degrees, log=False, seed=1, **kwargs) :
    """
    commenter ++++++
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best gammas and the associated test and train error
    best_err_te = []
    best_err_tr = []
    # vary degree
    for degree in degrees:
        #Call the function cross validation which returns the mean accuracy of the training and the test set
        acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, degree=degree, log=log, **kwargs)
        # store the mean error over the k-folds for each degree
        best_err_tr.append(1-acc_tr)
        best_err_te.append(1-acc_te)     
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(best_err_te)      
        
    return degrees[ind_best_degree], best_err_te[ind_best_degree], best_err_tr[ind_best_degree]