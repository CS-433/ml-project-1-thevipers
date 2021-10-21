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

def cross_validation_least_squares_GD(y, x, k_fold, gammas, degrees, seed=1, initial_w=None, max_iters=500) : 
    """
    Perform k-fold cross-validation to select the best model among various degrees with linear regression using gradient descent.
    For each degree, we compute the best learning rate gamma and the associated best test error.
    The degree which lead to the minimum test error is returned with the associated minimum test error and train error.
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the learning rates for gradient descent gammas
        * the various degrees we want to compare
        * the seed
        * the initial guess for the feature vector w initial_w
        * the maximal number of iterations for Gradient Descent max_iters
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
            err_tr_tmp = []
            err_te_tmp = []
            for k in range(k_fold):
                # get k'th subgroup in test, others in train
                te_indice = k_indices[k]
                tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
                tr_indice = tr_indice.reshape(-1)
                y_te = y[te_indice]
                y_tr = y[tr_indice]
                x_te = x[te_indice]
                x_tr = x[tr_indice]
                # form data with polynomial degree
                tx_tr = build_poly(x_tr, degree)
                tx_te = build_poly(x_te, degree)
                # gradient descent
                w, loss_tr = least_squares_GD(y_tr, tx_tr, initial_w, max_iters, gamma)
                # calculate the loss for train and test data
                y_pred_te = predict_labels(w, tx_te)
                loss_te = compute_loss(y_pred_te, y_te)
                err_tr_tmp.append(loss_tr)
                err_te_tmp.append(loss_te)
            # store the mean error over the k-folds for each lambda
            err_tr.append(np.mean(err_tr_tmp))
            err_te.append(np.mean(err_te_tmp))     
        # find the optimal lambda which lead to the minimum test error for the current degree
        # and store the optimal lambda, the minimum test error, and the train error for the same lambda 
        ind_gamma_opt = np.argmin(err_te)
        best_gammas.append(gammas[ind_gamma_opt])
        best_err_te.append(err_te[ind_gamma_opt])
        best_err_tr.append(err_tr[ind_gamma_opt])
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(best_err_te)      
        
    return degrees[ind_best_degree], best_err_te[ind_best_degree], best_err_tr[ind_best_degree]
    
def cross_validation_least_squares_SGD(y, x, k_fold, gammas, degrees, seed=1, initial_w=None, batch_size=1, max_iters=10000) : 
    """
    Perform k-fold cross-validation to select the best model among various degrees with linear regression using
    stochastic gradient descent.
    For each degree, we compute the best learning rate gamma and the associated best test error.
    The degree which lead to the minimal test error is returned with the associated minimum test error and train error.
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the learning rates for gradient descent gammas
        * the various degrees we want to compare
        * the seed
        * the initial guess for the feature vector w initial_w
        * the batch_size, which is the number of samples on which the new gradient is computed. If set to 1 it corresponds
        to SGD, if set to the full number of samples it is identical to least_squares_GD()
        * the maximal number of iterations for SGD max_iters
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
            err_tr_tmp = []
            err_te_tmp = []
            for k in range(k_fold):
                # get k'th subgroup in test, others in train
                te_indice = k_indices[k]
                tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
                tr_indice = tr_indice.reshape(-1)
                y_te = y[te_indice]
                y_tr = y[tr_indice]
                x_te = x[te_indice]
                x_tr = x[tr_indice]
                # form data with polynomial degree
                tx_tr = build_poly(x_tr, degree)
                tx_te = build_poly(x_te, degree)
                # SGD
                w, loss_tr = least_squares_SGD(y_tr, tx_tr, initial_w, batch_size, max_iters, gamma)
                # calculate the loss for train and test data
                y_pred_te = predict_labels(w, tx_te)
                loss_te = compute_loss(y_pred_te, y_te)
                err_tr_tmp.append(loss_tr)
                err_te_tmp.append(loss_te)
            # store the mean error over the k-folds for each lambda
            err_tr.append(np.mean(err_tr_tmp))
            err_te.append(np.mean(err_te_tmp))     
        # find the optimal lambda which lead to the minimum test error for the current degree
        # and store the optimal lambda, the minimum test error, and the train error for the same lambda 
        ind_gamma_opt = np.argmin(err_te)
        best_gammas.append(gammas[ind_gamma_opt])
        best_err_te.append(err_te[ind_gamma_opt])
        best_err_tr.append(err_tr[ind_gamma_opt])
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(best_err_te)      
        
    return degrees[ind_best_degree], best_err_te[ind_best_degree], best_err_tr[ind_best_degree]
    

def cross_validation_least_squares(y, x, k_fold, degrees, seed=1) : 
    """
    Perform k-fold cross-validation to select the best model among various degrees with least squares.
    The degree which lead to the minimal test error is returned with the associated minimum test error and train error.
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the various degrees we want to compare
        * the seed
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data for each degree
    err_tr = []
    err_te = []
    # vary degree
    for degree in degrees:  
        err_tr_tmp = []
        err_te_tmp = []
        for k in range(k_fold):
            # get k'th subgroup in test, others in train
            te_indice = k_indices[k]
            tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
            tr_indice = tr_indice.reshape(-1)
            y_te = y[te_indice]
            y_tr = y[tr_indice]
            x_te = x[te_indice]
            x_tr = x[tr_indice]
            # form data with polynomial degree
            tx_tr = build_poly(x_tr, degree)
            tx_te = build_poly(x_te, degree)
            # least squares
            w, loss_tr = least_squares(y_tr, tx_tr)
            # calculate the accuracy for train and test data
            y_pred_te = predict_labels(w, tx_te)
            loss_te = compute_loss(y_pred_te, y_te)
            err_tr_tmp.append(loss_tr)
            err_te_tmp.append(loss_te)
        # store the mean error over the k-folds for each degree
        err_tr.append(np.mean(err_tr_tmp))
        err_te.append(np.mean(err_te_tmp))     
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(err_te)      
        
    return degrees[ind_best_degree], err_te[ind_best_degree], err_tr[ind_best_degree]
    

def cross_validation_ridge_regression(y, x, k_fold, lambdas, degrees, seed=1) : 
    """
    Perform k-fold cross-validation to select the best model among various degrees with ridge regression.
    For each degree, we compute the best lambda and the associated best error.
    The degree which lead to the minimal test error is returned with the associated minimum test error and train error.
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the regularization parameters for ridge regression lambdas 
        * the various degrees we want to compare
        * the seed
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best lambdas and the associated test and train rmse
    best_lambdas = []
    best_err_te = []
    best_err_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the loss of training data and test data for each degree
        err_tr = []
        err_te = []
        # cross validation to find the best lambda for each degree
        for lambda_ in lambdas:
            err_tr_tmp = []
            err_te_tmp = []
            for k in range(k_fold):
                # get k'th subgroup in test, others in train
                te_indice = k_indices[k]
                tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
                tr_indice = tr_indice.reshape(-1)
                y_te = y[te_indice]
                y_tr = y[tr_indice]
                x_te = x[te_indice]
                x_tr = x[tr_indice]
                # form data with polynomial degree
                tx_tr = build_poly(x_tr, degree)
                tx_te = build_poly(x_te, degree)
                # ridge regression
                w, loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
                # calculate the loss for train and test data
                y_pred_te = predict_labels(w, tx_te)
                loss_te = compute_loss(y_pred_te, y_te)
                err_tr_tmp.append(loss_tr)
                err_te_tmp.append(loss_te)
            # store the mean error over the k-folds for each lambda
            err_tr.append(np.mean(err_tr_tmp))
            err_te.append(np.mean(err_te_tmp))     
        # find the optimal lambda which lead to the minimum test error for the current degree
        # and store the optimal lambda, the minimum test error, and the train error for the same lambda 
        ind_lambda_opt = np.argmin(err_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_err_te.append(err_te[ind_lambda_opt])
        best_err_tr.append(err_tr[ind_lambda_opt])
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(best_err_te)      
        
    return degrees[ind_best_degree], best_err_te[ind_best_degree], best_err_tr[ind_best_degree]
    
    
def cross_validation_logistic_regression(y, x, k_fold, gammas, degrees, seed=1, initial_w=None, batch_size=1, max_iters=10000) : 
    """
    Perform k-fold cross-validation to select the best model among various degrees with logistic regression using
    gradient descent or SGD. 
    For each degree, we compute the best learning rate gamma and the associated best test error.
    The degree which lead to the minimal test error is returned with the associated minimum test error and train error.
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the learning rates for gradient descent gammas
        * the various degrees we want to compare
        * the seed
        * the initial guess for the feature vector w initial_w
        * the batch_size, which is the number of samples on which the new gradient is computed. If set to 1 it corresponds
        to SGD, if set to the full number of samples it corresponds to GD.
        * the maximal number of iterations for SGD max_iters
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
            err_tr_tmp = []
            err_te_tmp = []
            for k in range(k_fold):
                # get k'th subgroup in test, others in train
                te_indice = k_indices[k]
                tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
                tr_indice = tr_indice.reshape(-1)
                y_te = y[te_indice]
                y_tr = y[tr_indice]
                x_te = x[te_indice]
                x_tr = x[tr_indice]
                # form data with polynomial degree
                tx_tr = build_poly(x_tr, degree)
                tx_te = build_poly(x_te, degree)
                # logistic regression
                w, _ = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma, batch_size)
                # calculate the accuracy for train and test data
                y_pred_tr = predict_logistic_labels(w, tx_tr)
                loss_tr = compute_loss(y_pred_tr, y_tr)
                y_pred_te = predict_logistic_labels(w, tx_te)
                loss_te = compute_loss(y_pred_te, y_te)
                err_tr_tmp.append(loss_tr)
                err_te_tmp.append(loss_te)
            # store the mean error over the k-folds for each lambda
            err_tr.append(np.mean(err_tr_tmp))
            err_te.append(np.mean(err_te_tmp))     
        # find the optimal lambda which lead to the minimum test error for the current degree
        # and store the optimal lambda, the minimum test error, and the train error for the same lambda 
        ind_gamma_opt = np.argmin(err_te)
        best_gammas.append(gammas[ind_gamma_opt])
        best_err_te.append(err_te[ind_gamma_opt])
        best_err_tr.append(err_tr[ind_gamma_opt])
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(best_err_te)      
        
    return degrees[ind_best_degree], best_err_te[ind_best_degree], best_err_tr[ind_best_degree]
    

def cross_validation_reg_logistic_regression(y, x, k_fold, gamma, lambdas, degrees, seed=1, initial_w=None, batch_size=1,
                                             max_iters=10000) : 
    """
    Perform k-fold cross-validation to select the best model among various degrees with regularized logistic regression.
    For each degree, we compute the best lambda and the associated best error.
    The degree which lead to the minimal test error is returned with the associated minimum test error and train error.
    Takes as input:
        * the targeted y
        * the sample matrix x
        * the number of folds k_fold
        * the learning rate for gradient descent gamma
        * the regularization parameter for regularized logistic regression lambda 
        * the various degrees we want to compare
        * the seed
        * the initial guess for the feature vector w initial_w
        * the batch_size, which is the number of samples on which the new gradient is computed. If set to 1 it corresponds
        to SGD, if set to the full number of samples it corresponds to GD.
        * the maximal number of iterations for SGD max_iters
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # for each degree, we compute the best lambdas and the associated test and train rmse
    best_lambdas = []
    best_err_te = []
    best_err_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the loss of training data and test data for each degree
        err_tr = []
        err_te = []
        # cross validation to find the best lambda for each degree
        for lambda_ in lambdas:
            err_tr_tmp = []
            err_te_tmp = []
            for k in range(k_fold):
                # get k'th subgroup in test, others in train
                te_indice = k_indices[k]
                tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
                tr_indice = tr_indice.reshape(-1)
                y_te = y[te_indice]
                y_tr = y[tr_indice]
                x_te = x[te_indice]
                x_tr = x[tr_indice]
                # form data with polynomial degree
                tx_tr = build_poly(x_tr, degree)
                tx_te = build_poly(x_te, degree)
                # ridge regression
                w, _ = reg_logistic_regression(y_tr, tx_tr, lambda_, initial_w, max_iters, gamma, batch_size)
                # calculate the accuracy for train and test data
                y_pred_tr = predict_labels(w, tx_tr)
                loss_tr = compute_loss(y_pred_tr, y_tr)
                y_pred_te = predict_labels(w, tx_te)
                loss_te = compute_loss(y_pred_te, y_te)
                err_tr_tmp.append(loss_tr)
                err_te_tmp.append(loss_te)
            # store the mean error over the k-folds for each lambda
            err_tr.append(np.mean(err_tr_tmp))
            err_te.append(np.mean(err_te_tmp))     
        # find the optimal lambda which lead to the minimum test error for the current degree
        # and store the optimal lambda, the minimum test error, and the train error for the same lambda 
        ind_lambda_opt = np.argmin(err_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_err_te.append(err_te[ind_lambda_opt])
        best_err_tr.append(err_tr[ind_lambda_opt])
    # find the degree which leads to the minimum test error
    ind_best_degree =  np.argmin(best_err_te)      
        
    return degrees[ind_best_degree], best_err_te[ind_best_degree], best_err_tr[ind_best_degree]
    
    

