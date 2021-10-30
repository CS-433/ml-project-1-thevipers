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
  

print("Preprocessing the data...")
.. = prepocess(..)

max_iter = 2000
gammas = [1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 0.001, 1e-05]
degrees = [3,3,3,3,3,3,3,3]

print("Finding the best weights and calculating predictions...")

# feature expansion
x_train[jet_num] = build_poly(x_train[jet_num], degrees[jet_num])
x_test[jet_num]  = build_poly(x_test[jet_num], degrees[jet_num])

# training the best model
w, loss = ...

# applying the w vector to the test data
test_results = predict_labels(w, x_test[jet_num], True)
final[indexes_test[jet_num]] = test_results
    
# creating the sumbission file
print("Creating the csv file for submission...")
create_csv_submission(ids_test, final, OUTPUT_PATH)

print("DONE!")

