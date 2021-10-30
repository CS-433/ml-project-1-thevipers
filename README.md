# Machine Learning Project 1

## General information 

The repository contains the code for Machine Learning course 2021 (CS-433) project 1 (Higgs Boson challenge: https://www.aicrowd.com/challenges/epfl-machine-learning-higgs) at EPFL. 

### Team members
The project is accomplished by the team `TheVipers` with members:

- Camille Frayssinhes: [@camillefrayssinhes](https://github.com/camillefrayssinhes)
- Assia Ouanaya: [@assiaoua](https://github.com/assiaoua)
- Theau Vannier: [@theauv](https://github.com/theauv)

With a Test Accuracy of xxxx we got the xxxxx-th place out of 228 teams.

### Data
The data `train.csv` and `test.csv` can be found in https://github.com/epfml/ML_course/tree/master/projects/project1/data, to run the code please download them and place them in a `data` folder. It is important to note that the initial output labels are {-1/1} but we remap them to {0/1} when loading the data.

### How to run the code
The project has been developed and test with `python3.8`.
The required library for running the models and training is `numpy1.20.1`.
The library for visualization is `matplotlib3.3.4`.

### How to reproduce the obtained results

The final results used to predict the test datasets for the final leaderbord place on AIcrowd are generated by running the function implemented in `run.py` (either in the last cell of the notebook, or by direcly running the `run.py` file by using the command `python3 run.py` in the terminal when you are located in the Scripts folder).
And the final results are saved in: `/data/results.csv`.

***
## Project architecture

### Helper functions

`proj1_helpers.py` : loading CSV training and test data, and creating CSV submission files.

### Processing data 

`preprocessing.py` : preprocessing training and test data for model training and prediction.


### Training data

`implementations.py` : the implementation of 6 methods to train the model : `least_squares_GD`, `least_squares_SGD`, `least_squares`, `ridge_regression`, `logistic_regression` and `reg_logistic_regression` and the associated functions needed to compute the associated losses.


### Selecting Model

`CV_modularised.py` : using cross-validation to test the accuracy of different models and searching for the best parameters(lambda, degree etc.) to obtain the best test accuracies.

`plot.py` : visualizing the training and test accuracy for different parameters, comparing the accuracy of different methods.


### Predicting test labels

`run.py` : generating the predictions for the test data using the selected best model. It is important to note that we are working with {0/1} output labels when training our methods but we give back {-1/1} labels for the output predictions in order to be compatible with the AIcrowd submission platform. 

### Notebook

`main.ipynb` : data exploration and preprocessing. Tuning the best parameters for the 6 methods and predicting the accuracy of all the methods through cross validation. Analysis and visualisation of the accuracy with different choices of parameters. Comparing the 6 methods and generating the label predictions for the test data.
 

### Report

`documents/report.pdf`: a 2-pages report of the project.


