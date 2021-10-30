# code for getting the final predictions for the test data

from Scripts.proj1_helpers import *
from Scripts.preprocessing import *
from Scripts.CV_modularised import *


# files need to be unziped before load
DATA_TRAIN_PATH = '../data/train.csv' 
DATA_TEST_PATH  = '../data/test.csv' 
OUTPUT_PATH = '../data/results.csv' 

print("Load the data from csv files...")
y_train, x_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
y_test, x_test, ids_test = load_csv_data(DATA_TEST_PATH)

# the final predictions
final = np.ones(y_test.shape)

print('TRAIN : Shape of y => {sy} \n\tShape of x => {sx}'.format(sy=y_train.shape, sx=x_train.shape))
print('TEST  : Shape of y => {sy} \n\tShape of x => {sx}'.format(sy=y_test.shape, sx=x_test.shape)) 
  
# splitting the data according to the jet number(0,1,2,3) and then according to the feature DER_mass_MMC
print("Splitting the train data...")
x_train = jet_dict(x_train)
y_train = labels_jet(y_train, indexes_train)

print("Splitting the test data...")
x_test, indexes_test = split_jet(x_test)
x_test, indexes_test = split_mass(x_test, indexes_test)
y_test = labels_jet(y_test, indexes_test)

print("Preprocessing the data...")
x_train = removeNaN(x_train)
x_test = removeNaN(x_test)

x_train = addingFeatures(x_train)
x_test = addingFeatures(x_test)


MAX_ITER = 2000
gammas = [1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 0.001, 1e-05]
degrees = [3,3,3,3,3,3,3,3]

print("Finding the best weights and calculating predictions...")
for jet_num in range (8):
    # replacing the nan values in columns where they are <100% present
    #replaceNaN(x_train[jet_num])
    
    # standardizing the data
    x_train[jet_num], mean, std = standardize(x_train[jet_num])
    x_test[jet_num],_,_  = standardize(x_test[jet_num],mean,std,True)
    
    # adding the offset term
    #x_train[jet_num] = np.c_[np.ones((y_train[jet_num].shape[0], 1)), x_train[jet_num]]
    #x_test[jet_num]  = np.c_[np.ones((y_test[jet_num].shape[0], 1)), x_test[jet_num]]
    
    # feature expansion
    x_train[jet_num] = build_poly(x_train[jet_num], degrees[jet_num])
    x_test[jet_num]  = build_poly(x_test[jet_num], degrees[jet_num])
    
    # initial w vector
    initial_w = np.random.randn(x_train[jet_num].shape[1])
    
    # training the model
    w, loss = logistic_regression(y_train[jet_num], x_train[jet_num], initial_w, MAX_ITER, gammas[jet_num])
    
    # applying the w vector to the test data
    test_results = predict_labels(w, x_test[jet_num], True)
    final[indexes_test[jet_num]] = test_results
    
# creating the sumbission file
print("Creating the csv file for submission...")
create_csv_submission(ids_test, final, OUTPUT_PATH)

print("DONE!")

