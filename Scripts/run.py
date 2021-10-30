# code for getting the final predictions for the test data

from Scripts.proj1_helpers import *
from Scripts.preprocessing import *
from Scripts.CV_modularised import *


def predicting_test_labels(method, degree, log, **kwargs) : 
    """
    Predict the output labels for the test dataset using the selected method.
    Takes as input:
        * the name of the method we want to use to predict the labels : method
        * the degree we use for polynomial expansion of the features : degree
        * the log boolean indicating if we are using a logistic method or not log
        * other arguments
    """
    # files need to be unziped before load
    DATA_TRAIN_PATH = 'Data/train.csv' 
    DATA_TEST_PATH  = 'Data/test.csv' 
    OUTPUT_PATH = 'Data/results.csv' 

    print("Load the data from csv files...")
    y_train, x_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
    y_test, x_test, ids_test = load_csv_data(DATA_TEST_PATH)

    print('TRAIN : Shape of y => {sy} \n\tShape of x => {sx}'.format(sy=y_train.shape, sx=x_train.shape))
    print('TEST  : Shape of y => {sy} \n\tShape of x => {sx}'.format(sy=y_test.shape, sx=x_test.shape)) 


    print("Preprocessing the data...")
    y_pred = np.zeros_like(y_test)
    x_train, x_test, y_train, _ = preprocess(x_train, x_test, y_train, y_pred, comment=False)

    print("Finding the best weights and calculating predictions...")

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

    # remapping the labels to {-1/1}    
    test_results[test_results==0] = -1

    # creating the submission file
    print("Creating the csv file for submission...")
    create_csv_submission(ids_test, test_results, OUTPUT_PATH)

    print("DONE!")

