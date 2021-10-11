# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def compute_mse(e):
    """Compute the mse for vector e."""
    mse = 1/2*np.mean(e**2)
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm using the least squares."""
    # Define parameter to store the last weight vector
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        grad, err = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
    # compute loss    
    loss = compute_mse(err)
    return w, loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err
 
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

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent using the least squares."""
    # Define parameters to store the last weight vector
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, err = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            
        loss = compute_mse(err)
    return w, loss

def least_squares(y, tx) :
    """Compute the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    # compute the weight vector and the loss using the MSE
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    loss = compute_mse(err)  
    return w, loss

def ridge_regression(y, tx, lambda_) :
    """Implement ridge regression."""
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    # compute the weight vector and the loss using the MSE
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    loss = compute_mse(err)   
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma) :
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma) :
    
    return w, loss


