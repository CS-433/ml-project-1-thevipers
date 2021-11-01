# -*- coding: utf-8 -*-
"""getting the final predictions for the test data"""

from proj1_helpers import *
from preprocessing import *
from crossvalidation import *


def predicting_test_labels(method, degree, log, split=True, **kwargs) : 
    """
    Predict the output labels for the test dataset using the selected method.
    Takes as input:
        * the name of the method we want to use to predict the labels : method
        * the degree we use for polynomial expansion of the features : degree
        * the log boolean indicating if we are using a logistic method or not log
        * other arguments
    """
    # files need to be unziped before load
    DATA_TRAIN_PATH = 'data/train.csv' 
    DATA_TEST_PATH  = 'data/test.csv' 
    OUTPUT_PATH = 'data/results.csv' 

    print("Load the data from csv files...")
    y_train, x_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
    y_test, x_test, ids_test = load_csv_data(DATA_TEST_PATH)

    print('TRAIN : Shape of y => {sy} \n\tShape of x => {sx}'.format(sy=y_train.shape, sx=x_train.shape))
    print('TEST  : Shape of y => {sy} \n\tShape of x => {sx}'.format(sy=y_test.shape, sx=x_test.shape)) 


    #Preprocessing of the data, training of the model and predicting the result :
    if(split) :
        test_results = train_with_jet(x_train, x_test, y_train, method, degree, log, **kwargs)
    else :
        test_results = train_without_jet(x_train, x_test, y_train, method, degree, log, **kwargs)

    # remapping the labels to {-1/1}    
    test_results[test_results==0] = -1

    # creating the submission file
    print("Creating the csv file for submission...")
    create_csv_submission(ids_test, test_results, OUTPUT_PATH)

    print("DONE!")

    
    
def train_with_jet(train_x, test_x, train_y, method, degree, log, **kwargs) :
    """
    Make the prediction of y using the splitting of the data according to the jet_num class of the sample (0, 1, {2, 3}).
    Takes as input:
        * the training sample matrix : train_x
        * the test sample matrix : test_x
        * the training labels : train_y
        * the name of the method we want to use to predict the labels : method
        * the degree we use for polynomial expansion of the features : degree
        * the log boolean indicating if we are using a logistic method or not log
        * other arguments
    """
    
    jet_dict_train = jet_dict(train_x)
    jet_dict_test = jet_dict(test_x)
    
    print("Finding the best weights and calculating predictions by splitting the data according to jet_num...")

    y_pred_test = np.zeros_like(test_x[:,0])
    for jet_num in range(jet_dict_train.shape[0]) :
            
            x_train_jet = train_x[jet_dict_train[jet_num]]
            x_test_jet = test_x[jet_dict_test[jet_num]]
            y_train_jet = train_y[jet_dict_train[jet_num]]

            x_train_jet, x_test_jet, y_train_jet= preprocess(x_train_jet, x_test_jet, y_train_jet)

            # form data with polynomial degree:
            x_train_jet = build_poly(x_train_jet, degree)
            x_test_jet = build_poly(x_test_jet, degree)
            
            #compute the best weights
            w, _ = method(y_train_jet, x_train_jet, **kwargs)

            #Compute y_pred[jet_dict]
            if log  :
                y_pred_test[jet_dict_test[jet_num]] = predict_logistic_labels(w, x_test_jet)
            else :
                y_pred_test[jet_dict_test[jet_num]] = predict_labels(w, x_test_jet)
    
    return y_pred_test

def train_without_jet(x_train, x_test, y_train, method, degree, log, **kwargs) :
    """
    Make the prediction of y without using the splitting of the data according to the jet_num class of the sample (0, 1, {2, 3}).
    Takes as input:
        * the training sample matrix : train_x
        * the test sample matrix : test_x
        * the training labels : train_y
        * the name of the method we want to use to predict the labels : method
        * the degree we use for polynomial expansion of the features : degree
        * the log boolean indicating if we are using a logistic method or not log
        * other arguments
    """
    
    print("Preprocessing the data...")
    x_train, x_test, y_train = preprocess(x_train, x_test, y_train, outliers_=True)

    print("Finding the best weights and calculating predictions without splitting according to jet_num...")

    # feature expansion
    x_train_deg = build_poly(x_train, degree)
    x_test_deg = build_poly(x_test, degree)

    # training the best model
    w, _ = method(y_train, x_train_deg, **kwargs)

    # applying the w vector to the test data to predict the labels {0/1}
    if (log) : 
        test_results = predict_logistic_labels(w, x_test_deg)
    else : 
        test_results = predict_labels(w, x_test_deg)
        
    return test_results
    
    
################
# create final submission
predicting_test_labels(ridge_regression, 8, log=False, split = False, lambda_= 0.00001)    
