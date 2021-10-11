# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def compute_mse(e):
    """Compute the Mean Square Error.
    Take as input the error vector e. 
    """
    mse = 1/2*np.mean(e**2)
    return mse

def compute_gradient(y, tx, w):
    """
    Compute the gradient with respect to w.
    Takes as input the targeted y, the sample matrix tx and the feature vector w.
    This function is used when solving gradient based method, such that least_squares_GD() and least_squares_SGD().
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
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
    """
    Compute an estimated solution of the problem y = tx @ w and the associated error using Stochastic Gradient Descent. 
    Takes as input:
        * the targeted y
        * the sample matrix tx
        * the initial guess for w initial_w
        * the batch_size, which is the number of samples on which the new gradient is computed. If set to 1 it corresponds
        to Stochastic Gradient Descent, if set to the full number of samples it is identifical to least_squares_GD().
        * the maximal number of iterations for Gradient Descent max_iters
        * the learning rate gamma
    """
    # Define parameters to store the last weight vector
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, err = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            
        loss = compute_mse(err)
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
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    loss = compute_mse(err)  
    return w, loss

def ridge_regression(y, tx, lambda_) :
    """
    Compute an estimated solution of the problem y = tx @ w , and the associated error. Note that this method
    is a variant of least_square() but with a regularization term lambda_. 
    This method is equivalent to the minimization problem of finding w such that |y-tx@w||^2 + lambda_*||w||^2 is minimal. 
    The error is the mean square error of the targeted y and the solution produced by the least square function.
    Takes as input the targeted y, the sample matrix X and the regularization term lambda_.
    """
       
    aI = lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    # compute the weight vector and the loss using the MSE
    w = np.linalg.solve(a, b)
    err = y - tx.dot(w)
    loss = compute_mse(err)   
    return w, loss

def sigmoid(t):
    """
    Apply the sigmoid function on t.
    """
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood.
    Takes as input the targeted y, the sample matrix tx and the feature fector w.
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma) :
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma) :
    
    return w, loss


