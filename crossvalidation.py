# -*- coding: utf-8 -*-
"""cross validation"""
import numpy as np
from implementations import *
from helpers import *

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

def cross_validation_least_squares_GD(y, x, k_fold, gammas, degrees, seed=1, initial_w=None, max_iters=10000)
    """
    Perform k-fold cross-validation to select the best model among various degrees with linear regression using gradient descent.
    For each degree, we compute the best learning rate gamma and the associated best test rmse.
    The degree which lead to the minimum test rmse is returned with the associated minimum test rmse and train rmse.
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
    # for each degree, we compute the best gammas and the associated test and train rmse
    best_gammas = []
    best_rmses_te = []
    best_rmses_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the loss of training data and test data for each degree
        rmse_tr = []
        rmse_te = []
        # cross validation to find the best learning rate gamma for each degree
        for gamma in gammas:
            rmse_tr_tmp = []
            rmse_te_tmp = []
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
                w, _ = least_squares_GD(y_tr, tx_tr, initial_w, max_iters, gamma)
                # calculate the loss for train and test data
                loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
                loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
                rmse_tr_tmp.append(loss_tr)
                rmse_te_tmp.append(loss_te)
            # store the mean rmse over the k-folds for each lambda
            rmse_tr.append(np.mean(rmse_tr_tmp))
            rmse_te.append(np.mean(rmse_te_tmp))     
        # find the optimal lambda which lead to the minimum test rmse for the current degree
        # and store the optimal lambda, the minimum test rmse, and the train rmse for the same lambda 
        ind_gamma_opt = np.argmin(rmse_te)
        best_gammas.append(gammas[ind_gamma_opt])
        best_rmses_te.append(rmse_te[ind_gamma_opt])
        best_rmses_tr.append(rmse_tr[ind_gamma_opt])
    # find the degree which leads to the minimum test rmse
    ind_best_degree =  np.argmin(best_rmses)      
        
    return degrees[ind_best_degree], best_rmses_te[ind_best_degree], best_rmses_tr[ind_best_degree]
    
def cross_validation_least_squares_SGD(y, x, k_fold, gammas, degrees, seed=1, initial_w=None, batch_size=1, max_iters=10000)
    """
    Perform k-fold cross-validation to select the best model among various degrees with linear regression using
    stochastic gradient descent.
    For each degree, we compute the best learning rate gamma and the associated best test rmse.
    The degree which lead to the minimal test rmse is returned with the associated minimum test rmse and train rmse.
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
    # for each degree, we compute the best gammas and the associated test and train rmse
    best_gammas = []
    best_rmses_te = []
    best_rmses_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the loss of training data and test data for each degree
        rmse_tr = []
        rmse_te = []
        # cross validation to find the best learning rate gamma for each degree
        for gamma in gammas:
            rmse_tr_tmp = []
            rmse_te_tmp = []
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
                w, _ = least_squares_SGD(y_tr, tx_tr, initial_w, batch_size, max_iters, gamma)
                # calculate the loss for train and test data
                loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
                loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
                rmse_tr_tmp.append(loss_tr)
                rmse_te_tmp.append(loss_te)
            # store the mean rmse over the k-folds for each lambda
            rmse_tr.append(np.mean(rmse_tr_tmp))
            rmse_te.append(np.mean(rmse_te_tmp))     
        # find the optimal lambda which lead to the minimum test rmse for the current degree
        # and store the optimal lambda, the minimum test rmse, and the train rmse for the same lambda 
        ind_gamma_opt = np.argmin(rmse_te)
        best_gammas.append(gammas[ind_gamma_opt])
        best_rmses_te.append(rmse_te[ind_gamma_opt])
        best_rmses_tr.append(rmse_tr[ind_gamma_opt])
    # find the degree which leads to the minimum test rmse
    ind_best_degree =  np.argmin(best_rmses)      
        
    return degrees[ind_best_degree], best_rmses_te[ind_best_degree], best_rmses_tr[ind_best_degree]
    

def cross_validation_least_squares(y, x, k_fold, degrees, seed=1)
    """
    Perform k-fold cross-validation to select the best model among various degrees with least squares.
    The degree which lead to the minimal test rmse is returned with the associated minimum test rmse and train rmse.
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
    rmse_tr = []
    rmse_te = []
    # vary degree
    for degree in degrees:  
        rmse_tr_tmp = []
        rmse_te_tmp = []
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
            w, _ = least_squares(y_tr, tx_tr)
            # calculate the loss for train and test data
            loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
            loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        # store the mean rmse over the k-folds for each degree
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))     
    # find the degree which leads to the minimum test rmse
    ind_best_degree =  np.argmin(rmse_te)      
        
    return degrees[ind_best_degree], rmse_te[ind_best_degree], rmse_tr[ind_best_degree]
    

def cross_validation_ridge_regression(y, x, k_fold, lambdas, degrees, seed=1)
    """
    Perform k-fold cross-validation to select the best model among various degrees with ridge regression.
    For each degree, we compute the best lambda and the associated best rmse.
    The degree which lead to the minimal test rmse is returned with the associated minimum test rmse and train rmse.
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
    best_rmses_te = []
    best_rmses_tr = []
    # vary degree
    for degree in degrees:
        # define lists to store the loss of training data and test data for each degree
        rmse_tr = []
        rmse_te = []
        # cross validation to find the best lambda for each degree
        for lambda_ in lambdas:
            rmse_tr_tmp = []
            rmse_te_tmp = []
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
                w, _ = ridge_regression(y_tr, tx_tr, lambda_)
                # calculate the loss for train and test data
                loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
                loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))
                rmse_tr_tmp.append(loss_tr)
                rmse_te_tmp.append(loss_te)
            # store the mean rmse over the k-folds for each lambda
            rmse_tr.append(np.mean(rmse_tr_tmp))
            rmse_te.append(np.mean(rmse_te_tmp))     
        # find the optimal lambda which lead to the minimum test rmse for the current degree
        # and store the optimal lambda, the minimum test rmse, and the train rmse for the same lambda 
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_rmses_te.append(rmse_te[ind_lambda_opt])
        best_rmses_tr.append(rmse_tr[ind_lambda_opt])
    # find the degree which leads to the minimum test rmse
    ind_best_degree =  np.argmin(best_rmses)      
        
    return degrees[ind_best_degree], best_rmses_te[ind_best_degree], best_rmses_tr[ind_best_degree]
    
    
def cross_validation_logistic_regression()

def cross_validation_reg_logistic_regression()

    

