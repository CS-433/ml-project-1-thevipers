# -*- coding: utf-8 -*-
"""ML methods."""
import numpy as np
from proj1_helpers import *

def compute_mse(y, tx, w):
    """
    Compute the Mean Square Error.
    Take as input the targeted y, the sample matrix tx and the feature vector w. 
    """
    e = y - tx.dot(w)
    mse = 1/2*np.mean(e**2)
    return mse

def compute_gradient(y, tx, w):
    """
    Compute the gradient with respect to w.
    Takes as input the targeted y, the sample matrix tx and the feature vector w.
    This function is used when solving gradient based method, such that least_squares_GD() and least_squares_SGD().
    """
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def least_squares_GD(y, tx, initial_w=None, max_iters=10000, gamma=0.01):
    """
    Compute an estimated solution of the problem y = tx @ w and the associated error using Gradient Descent. 
    This method is equivalent to the minimization problem of finding w such that |y-tx@w||^2 is minimal. Note that 
    this method may output a local minimum.
    Takes as input:
        * the targeted y
        * the sample matrix tx
        * the initial guess for w initial_w
        * the maximal number of iterations for Gradient Descent max_iters
        * the learning rate gamma
    """
    if np.all(initial_w == None): initial_w = np.zeros(tx.shape[1])
    # Define parameters 
    w = initial_w
    threshold = 1e-8
    losses = []
    for n_iter in range(max_iters):
        # compute gradient
        grad, err = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
        print('w : ', w)
    # compute loss  
    y_pred = predict_labels(w, tx)
    loss = compute_loss(y_pred, y)
    print('loss', loss)
    losses.append(loss)
    if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2]) < threshold):
                return w, losses[-1]
    return w, losses[-1]
 
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]    

def least_squares_SGD(y, tx, initial_w=None, batch_size=1, max_iters=10000, gamma=0.01):
    """
    Compute an estimated solution of the problem y = tx @ w and the associated error using Stochastic Gradient Descent. 
    Takes as input:
        * the targeted y
        * the sample matrix tx
        * the initial guess for w initial_w
        * the batch_size, which is the number of samples on which the new gradient is computed. If set to 1 it corresponds
        to Stochastic Gradient Descent, if set to the full number of samples it is identical to least_squares_GD()
        * the maximal number of iterations for SGD max_iters
        * the learning rate gamma
    """
    if np.all(initial_w == None): initial_w = np.random.rand(tx.shape[1])
    # Define parameters to store the last weight vector
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, e = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
         
        # compute loss  
        y_pred = predict_labels(w, tx)
        loss = compute_loss(y_pred, y)
    return w, loss

def least_squares(y, tx) :
    """
     Compute a closed-form solution of the problem y = tx @ w, and the associated error. This method is equivalent 
     to the minimization problem of finding w such that |y-tx@w||^2 is minimal.
     Note that this methods provides the global optimum.
     The error is the mean square error of the targeted y and the solution produced by the least square function.
     Takes as input the targeted y, and the sample matrix tx.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    # compute the weight vector and the loss using the MSE
    w = np.linalg.lstsq(a, b, rcond=None)[0]
    # compute loss  
    y_pred = predict_labels(w, tx)
    loss = compute_loss(y_pred, y) 
    return w, loss


def ridge_regression(y, tx, lambda_=0.1) :
    """
    Compute an estimated solution of the problem y = tx @ w , and the associated error. Note that this method
    is a variant of least_square() but with a regularization term lambda_. 
    This method is equivalent to the minimization problem of finding w such that |y-tx@w||^2 + lambda_*||w||^2 is minimal. 
    The error is the mean square error of the targeted y and the solution produced by the least square function.
    Takes as input the targeted y, the sample matrix X and the regularization term lambda_.
    """
       
    aI = 2*tx.shape[0]*lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    # compute the weight vector and the loss using the MSE
    w = np.linalg.solve(a, b)
    # compute loss  
    y_pred = predict_labels(w, tx)
    loss = compute_loss(y_pred, y)  
    return w, loss


def calculate_logistic_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood.
    Takes as input the targeted y, the sample matrix tx and the feature fector w.
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_logistic_gradient(y, tx, w):
    """
    Compute the gradient of the loss with respect to w.
    Takes as input the targeted y, the sample matrix tx and the feature vector w. 
    This function is used when solving gradient based method, such that logistic_regression() and reg_logistic_regression().
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma=0.01):
    """
    Do one step of gradient descent using logistic regression.
    Takes as input the targeted y, the sample matrix tx, the feature w and the learning rate gamma.
    Return the loss and the updated feature vector w.
    """
    grad = calculate_logistic_gradient(y, tx, w)
    #loss = calculate_logistic_loss(y, tx, w)
    w = w - gamma * grad
    #return w, loss
    return w


def logistic_regression(y, tx, initial_w = None, max_iters=10000, gamma=0.01, batch_size=1) :
    """
    Compute an estimated solution of the problem y = sigmoid(tx @ w) and the associated error using Gradient Descent or SGD. 
    This method is equivalent to the minimization problem of finding w such that the negative log likelihood is minimal.
    Note that this method may output a local minimum.
    Takes as input:
        * the targeted y
        * the sample matrix tx
        * the initial guess for w initial_w
        * the maximal number of iterations for Gradient Descent max_iters
        * the learning rate gamma
        * the batch_size, which is the number of samples on which the new gradient is computed.
        If set to 1 (by default) it corresponds to SGD, it set to the full number of samples it is Gradient Descent.
    """
    # init parameters
    if np.all(initial_w == None): initial_w = np.zeros(tx.shape[1])
    threshold = 1e-8
    losses = []
    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # get loss and update w.
            #w, _ = learning_by_gradient_descent(y_batch, tx_batch, w, gamma)
            w = learning_by_gradient_descent(y_batch, tx_batch, w, gamma)
            # compute loss  
            y_pred = predict_logistic_labels(w, tx)
            loss = compute_loss(y_pred, y) 
            # converge criterion
            if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2]) < threshold) :
                return w, losses[-1] 
    return w, losses[-1]


def learning_by_penalized_gradient(y, tx, w, gamma=0.01, lambda_=0.1):
    """
    Compute one step of gradient descent for regularized logistic regression.
    Takes as input the targeted y, the sample matrix tx, the feature vector w, the learning rate gamma and the
    regularization term lambda_.
    Return the loss and the updated feature vector w.
    """
    loss = calculate_logistic_loss(y, tx, w) + lambda_*np.squeeze(w.T.dot(w))
    gradient = calculate_logistic_gradient(y, tx, w) + 2*lambda_*w
    w = w - gamma * gradient
    return w, loss

def reg_logistic_regression(y, tx, lambda_ , initial_w=None, max_iters=10000, gamma=0.01, batch_size=1) :
    """
    Compute an estimated solution of the problem y = sigmoid(tx @ w) and the associated error using Gradient Descent. 
    Note that this method is a variant of logistic_regression() but with an added regularization term lambda_. 
    This method is equivalent to the minimization problem of finding w such that the negative log likelihood is minimal.
    Note that this method may output a local minimum.
    Takes as input:
        * the targeted y
        * the sample matrix tx
        * the regularization term lambda_
        * the initial guess for w initial_w
        * the maximal number of iterations for Gradient Descent max_iters
        * the learning rate gamma
        * the batch_size, which is the number of samples on which the new gradient is computed. If set to 1 it corresponds
        to SGD, if set to the full number of samples it is Gradient Descent.
    """
    # init parameters
    if np.all(initial_w == None): initial_w = np.zeros(tx.shape[1])
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            w, l_ = learning_by_penalized_gradient_descent(y_batch, tx_batch, w, gamma, lambda_)
            # converge criterion
            loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
            losses.append(loss)
            if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2]) < threshold):
                return w, losses[-1]
    return w, losses[-1]


