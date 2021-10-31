# -*- coding: utf-8 -*-
"""cross validation"""
import numpy as np
from implementations import *
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


####################
#Classical cross-validation and a cross-validation that takes into account the jet_num :

def cross_validation(y, x, k_indices, k_fold, method, degree=1, lambda_=None, gamma=None, log=False, **kwargs):
    """
    Perform k-fold cross-validation to select the best model among various degrees with any of the 6 different
    methods implemented.
    Return the training and test accuracies obtained by cross validation for any of the 6 different methods. 
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the indices of the test set k_indices
        * the number of folds k_fold
        * the method we want to use method
        * the various degrees we want to compare degree
        * the tuning parameter for regularized methods lambda_
        * the learning rates for methods using gradient descent gamma
        * the log boolean indicating if we are using a logistic method or not log
        * other arguments 
    """
    
    acc_train=[]
    acc_test=[]
    
    # cross-validation repeated for each fold (at each iteration the test set is different)
    for k in range(k_fold) :
        
        # split the train and test sets
        test_i = k_indices[k]
        train_i = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        train_i = train_i.reshape(-1)
        test_x = x[test_i, :]
        test_y = y[test_i]
        train_x = x[train_i, :]
        train_y = y[train_i]

        # form data with polynomial degree:
        train_x = build_poly(train_x, degree)
        test_x = build_poly(test_x, degree)
        # compute the weights
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

        #Compute y_pred
        if log  :
            y_pred_train = predict_logistic_labels(w, train_x)
            y_pred_test = predict_logistic_labels(w, test_x)
        else :
            y_pred_train = predict_labels(w, train_x)
            y_pred_test = predict_labels(w, test_x)
            
        # calculate the accuracy for train and test data
        acc_train.append(compute_accuracy(train_y, y_pred_train))
        acc_test.append(compute_accuracy(test_y, y_pred_test))
    
    # average the accuracies over the 'k_fold' folds
    acc_tr = np.mean(acc_train)
    acc_te = np.mean(acc_test)
    
    return acc_tr, acc_te

def cross_validation_jet(y, x, k_indices, k_fold, method, degree=1, lambda_=None, gamma=None, log=False, **kwargs):
    """
    Perform the same thing as the cross_validation above, except it will split the data into 3 sub datasets 
    (according to their jet_num value). 
    It will also process the data (for instance remove the meaningless features for each of these sub-datasets) before doing 
    cross validation.
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the indices of the test set k_indices
        * the number of folds k_fold
        * the method we want to use method
        * the various degrees we want to compare degree
        * the tuning parameter for regularized methods lambda_
        * the learning rates for methods using gradient descent gamma
        * the log boolean indicating if we are using a logistic method or not log
        * other arguments
    """     
        
    acc_train=[]
    acc_test=[]

    # cross-validation repeated for each fold (at each iteration the test set is different)
    for k in range(k_fold) :
    
        # split the train and test sets
        test_i = k_indices[k]
        train_i = (np.delete(k_indices, k, axis=0)).reshape(-1)
        test_x = x[test_i, :]
        test_y = y[test_i]
        train_x = x[train_i, :]
        train_y = y[train_i]
        
        jet_dict_train = jet_dict(train_x)
        jet_dict_test = jet_dict(test_x)
        
        y_pred_train = np.zeros_like(train_y)
        y_pred_test = np.zeros_like(test_y)
        
        for jet_num in range(jet_dict_train.shape[0]) :
            
            x_train_jet = train_x[jet_dict_train[jet_num]]
            x_test_jet = test_x[jet_dict_test[jet_num]]
            y_train_jet = train_y[jet_dict_train[jet_num]]

            x_train_jet, x_test_jet, y_train_jet= preprocess(x_train_jet, x_test_jet, y_train_jet)

            # form data with polynomial degree:
            x_train_jet = build_poly(x_train_jet, degree)
            x_test_jet = build_poly(x_test_jet, degree)
            
            # compute the weights
            if(lambda_==None) :
                if(gamma==None) :
                    w, loss = method(y_train_jet, x_train_jet, **kwargs)
                else :
                    w, loss = method(y_train_jet, x_train_jet, gamma=gamma, **kwargs)
            else :
                if(gamma==None) :
                    w, loss = method(y_train_jet, x_train_jet, lambda_=lambda_, **kwargs)
                else :
                    w, loss = method(y_train_jet, x_train_jet, lambda_=lambda_, gamma=gamma, **kwargs)

            #Compute y_pred[jet_dict]
            if log  :
                y_pred_train[jet_dict_train[jet_num]] = predict_logistic_labels(w, x_train_jet)
                y_pred_test[jet_dict_test[jet_num]] = predict_logistic_labels(w, x_test_jet)
            else :
                y_pred_train[jet_dict_train[jet_num]] = predict_labels(w, x_train_jet)
                y_pred_test[jet_dict_test[jet_num]] = predict_labels(w, x_test_jet)
            
        # calculate the accuracy for train and test data
        acc_train.append(compute_accuracy(train_y, y_pred_train))
        acc_test.append(compute_accuracy(test_y, y_pred_test))

    # average the accuracies over the 'k_fold' folds
    acc_tr = np.mean(acc_train)
    acc_te = np.mean(acc_test)
    
    return acc_tr, acc_te


####################
#The following functions use the cross validations function to tune the parameters :

def tune_best_one(y, x, k_fold, method, seed, params, name='degree', log=False, **kwargs) :
    """
    Tune one of the following parameters : degree, lambda or gamma
    This function can take any of the six methods implemented and tune the parameter we want to optimize.
    It returns the optimal parameter as well as the test accuracies and the training accuracies for all the parameters. 
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the method we want to use : method
        * the seed : seed
        * the parameters we want to compare params
        * the name of the parameters we want to compare : name. This could be either 'degree', 'lambda', or 'gamma'.
        * the log boolean indicating if we are using a logistic method or not log
        * other arguments 
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best gammas and the associated test and train error
    accs_te = []
    accs_tr = []
    # vary degree
    for param in params:
        # call the function cross validation which returns the mean accuracy of the training and the test set
        if(name== 'degree') :
            acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, degree=param, log=log, **kwargs)
        elif(name== 'lambda') :
            acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, lambda_=param, log=log, **kwargs)
        elif(name== 'gamma'):
            acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, gamma=param, log=log, **kwargs)
        else :
            raise NameError(name, ' is not a name of one of the tunable parameters')
        # store the mean accuracy over the k-folds for each degree
        accs_tr.append(acc_tr)
        accs_te.append(acc_te)     
    # find the degree which leads to the maximum accuracy
    ind_best_param =  np.argmax(accs_te)      
        
    return params[ind_best_param], accs_te, accs_tr
    
    

####################################################################################################################
# The following functions aim to tune parameters simultaneously. It can be better in general, but are a bit heavy in term of 
# computational cost.

def tune_best_deg_lam_gam(y, x, k_fold, method, degrees, lambdas, gammas, log=False, seed=1, **kwargs) : 
    """
    Tune the three hyper-parameters : degree, lambda and gamma. 
    This function can take any of the six methods implemented and tune the parameters we want to optimize.
    It returns the optimal set of parameters as well as the best associated test accuracy and the associated training accuracy. 
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the method we want to use : method
        * the various degrees we want to compare : degrees
        * the regularization parameters for regularization lambdas 
        * the learning rates for gradient descent gammas
        * the log boolean indicating if we are using a logistic method or not log
        * the seed : seed
        * other arguments 
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best gammas and the associated test and train accuracy
    best_lambdas = []
    best_gammas = []
    best_acc_te = []
    best_acc_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the accuracy of training data and test data for each degree
        acc_tr_deg = []
        acc_te_deg = []
        best_gammas_tmp = []
        #vary lambda
        for lambda_ in lambdas:
            # define lists to store the accuracy of training data and test data for each degree
            acc_tr_gam = []
            acc_te_gam = []
            #vary lambda
            for gamma in gammas:
                #Call the function cross validation which returns the mean accuracy of the training and the test set
                acc_tr, acc_te = cross_validation(y, x, k_indices=k_indices, k_fold=k_fold, method=method, degree=degree,
                                                                        lambda_=lambda_, gamma=gamma, log=log, **kwargs)
                # store the mean error over the k-folds for each lambda
                acc_tr_gam.append(acc_tr)
                acc_te_gam.append(acc_te)  
                
            # find the optimal gamma which lead to the maximum test accuracy for the current degree and lambda
            # and store the optimal gamma, the maximum test accuracy, and the train accuracy for each lambdas
            ind_gam_opt = np.argmax(acc_te_gam)
            best_gammas_tmp.append(gammas[ind_gam_opt])
            acc_te_deg.append(acc_te_gam[ind_gam_opt])
            acc_tr_deg.append(acc_tr_gam[ind_gam_opt])
            
        # find the optimal lambda which lead to the maximum test accuracy for the current degree
        # and store the optimal lambda, the maximum test accuracy, and the train accuracy for the same lambda 
        ind_lambda_opt = np.argmax(acc_te_deg)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_gammas.append(best_gammas_tmp[ind_lambda_opt])
        best_acc_te.append(acc_te_deg[ind_lambda_opt])
        best_acc_tr.append(acc_tr_deg[ind_lambda_opt])
            
    # find the degree which leads to the maximum test accuracy
    ind_best_degree =  np.argmax(best_acc_te)
        
    return degrees[ind_best_degree], best_gammas[ind_best_degree], best_lambdas[ind_best_degree], best_acc_te[ind_best_degree], best_acc_tr[ind_best_degree]

    
    

def tune_best_deg_gam(y, x, k_fold, method, degrees, gammas, log=False, seed=1, **kwargs) : 
    """
    Tune the two hyper-parameters : degree and gamma. 
    This function can take any of the six methods implemented and tune the parameters we want to optimize, but we use it for
    methods which have only gamma and degree parameters to tune (not for the methods which have an additional regularization
    parameter lambda to tune).
    It returns the optimal set of parameters as well as the best associated test accuracy and the associated training accuracy. 
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the method we want to use : method
        * the various degrees we want to compare : degrees
        * the learning rates for gradient descent gammas
        * the log boolean indicating if we are using a logistic method or not log
        * the seed : seed
        * other arguments 
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best gammas and the associated test and train accuracy
    best_gammas = []
    best_acc_te = []
    best_acc_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the accuracy of training data and test data for each degree
        acc_tr_deg = []
        acc_te_deg = []
        # vary gamma
        for gamma in gammas:
            # call the function cross validation which returns the mean accuracy of the training and the test set
            acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, degree=degree, gamma=gamma, log=log, **kwargs)
            # store the mean accuracy over the k-folds for each lambda
            acc_tr_deg.append(acc_tr)
            acc_te_deg.append(acc_te)     
        # find the optimal lambda which lead to the maximum test accuracy for the current degree
        # and store the optimal lambda, the maximum test accuracy, and the train accuracy for the same lambda 
        ind_gamma_opt = np.argmin(acc_te_deg)
        best_gammas.append(gammas[ind_gamma_opt])
        best_acc_te.append(acc_te_deg[ind_gamma_opt])
        best_acc_tr.append(acc_tr_deg[ind_gamma_opt])
    # find the degree which leads to the maximum test accuracy
    ind_best_degree =  np.argmax(best_acc_te)

    return degrees[ind_best_degree], best_gammas[ind_best_degree], best_acc_te[ind_best_degree], best_acc_tr[ind_best_degree]



def tune_best_deg_lam(y, x, k_fold, method, degrees, lambdas, log=False, seed=1, **kwargs) : 
    """
    Tune the two hyper-parameters : degree and lambda. 
    This function can take any of the six methods implemented and tune the parameters we want to optimize, but we use it for
    methods which have only lambda and degree parameters to tune (not for the methods which have an additional learning rate
    parameter gamma to tune). 
    It returns the optimal set of parameters as well as the best associated test accuracy and the associated training accuracy. 
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the method we want to use : method
        * the various degrees we want to compare : degrees
        * the regularization parameters for regularization lambdas 
        * the log boolean indicating if we are using a logistic method or not log
        * the seed : seed
        * other arguments 
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best gammas and the associated test and train accuracy
    best_lambdas = []
    best_acc_te = []
    best_acc_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the loss of training data and test data for each degree
        acc_tr_deg = []
        acc_te_deg = []
        # vary lambda
        for lambda_ in lambdas:
            # call the function cross validation which returns the mean accuracy of the training and the test set
            acc_tr, acc_te = cross_validation(y, x, k_indices=k_indices, k_fold=k_fold, method=method, degree=degree,
                                                                                      lambda_=lambda_, log=log, **kwargs)
            # store the mean accuracy over the k-folds for each lambda
            acc_tr_deg.append(acc_tr)
            acc_te_deg.append(acc_te)     
        # find the optimal lambda which lead to the maximum test accuracy for the current degree
        # and store the optimal lambda, the maximum test accuracy, and the train accuracy for the same lambda 
        ind_lambda_opt = np.argmax(acc_te_deg)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_acc_te.append(acc_te_deg[ind_lambda_opt])
        best_acc_tr.append(acc_tr_deg[ind_lambda_opt])
    # find the degree which leads to the maximum test accuracy
    ind_best_degree =  np.argmax(best_acc_te)      
        
    return degrees[ind_best_degree], best_lambdas[ind_best_degree], best_acc_te[ind_best_degree], best_acc_tr[ind_best_degree]



def tune_best_deg(y, x, k_fold, method, degrees, log=False, seed=1, **kwargs) :
    """
    Tune the the following hyper-parameter : degree. 
    This function can take any of the six methods implemented and tune the parameter we want to optimize (i.e. degree).
    It returns the optimal degree as well as the best associated test accuracy and the associated training accuracy. 
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the method we want to use : method
        * the various degrees we want to compare : degrees
        * the log boolean indicating if we are using a logistic method or not log
        * the seed : seed
        * other arguments 
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best gammas and the associated test and train accuracy
    best_acc_te = []
    best_acc_tr = []
    # vary degree
    for degree in degrees:
        # call the function cross validation which returns the mean accuracy of the training and the test set
        acc_tr, acc_te = cross_validation(y, x, k_indices, k_fold, method, degree=degree, log=log, **kwargs)
        # store the mean accuracy over the k-folds for each degree
        best_acc_tr.append(acc_tr)
        best_acc_te.append(acc_te)     
    # find the degree which leads to the maximum test accuracy
    ind_best_degree =  np.argmax(best_acc_te)      
        
    return degrees[ind_best_degree], best_acc_te[ind_best_degree], best_acc_tr[ind_best_degree]