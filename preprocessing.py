# -*- coding: utf-8 -*-
"""
GENERAL DESCRIPTION :
The pre-processing aims to modify the raw data into an easily analyzed data set
For instance we can delete the 'useless' features (column of zeros, correlated values,...), standarize,  deal with missing values, NaN, ...
"""
import numpy as np



""" This function aims to add a first column of 1 in X (usually called the offset) """
def offset(x) :
    new_x = np.c_[np.ones(x.shape[0]), x]
    return new_x


""" Count the missing values of each feature and delete columns (features) above a certain treshold of missing values
    If the percentage is below the treshold, missing values will be kept, but set to the mean of the column
    Becareful : *missing value = -999* """
#QUESTION : Est ce que les 0 sont aussi des missing values ????
def missing_values(tx_train, tx_test, treshold=0.9) :
    
    new_train = tx_train
    new_test = tx_test
    
    for col in range(tx_train.shape[1]) :
        
        #We calculate what percentage of missing values there is in each column :
        miss_vals = np.sum(tx_train[:,col]==-999)/999
        percentage = miss_vals/tx_train[:,0].shape
        
        #We will change every missing value in the column to the mean of the column (arbitrary choice) :
        feature = new_train[:, col]
        where = np.where(feature!=-999, True, False)
        mean = np.mean(feature, where=where)
        feature[feature==-999] = mean
        new_train[:,col] = feature
        
        #If the percentage of missing values in a column is above a treshold, we will just completely remove the column, 
        #because the column is consider useless for the prediction
        if(percentage >= treshold) :
            new_train = np.delete(new_train, col, axis=1)
            new_test = np.delete(new_test, col, axis=1)
            
    return new_train, new_test


""" If we want to standardize a(n) value/array using the method x_standardized = (x - mean) / std """
#CHECKER SI BIEN CA QUI FAUT FAIRE
def standardize(x) :
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    
    x_std = (x - mean)/std
    
    return x_std


#???? Do we need it ????
def correlation() :
    
    return 0


""" Outliers are aberrant values, usually due to some errors in the experiment or from the material, 
    but those values are usually not relevant and may impact the prediction. One way to process the data is to get rid of 
    those aberrant values """
def outliers(x) :
    
    new_x=x
    for i in range(x.shape[1]) :
        idx = findOutliers(x[:, i])
        np.delete(new_x, idx, axis=0)
    
    return new_x
    

""" Used in outliers """
def findOutliers(x, outlierConstant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    outliers_idx = []
    for i, y in enumerate(a.tolist()):
        if y <= quartileSet[0] or y >= quartileSet[1]:
            outliers_idx.append(i)
    return outliers_idx

""" feature engineering by extending the dataset into polynomial dataset : y = 1 + x + x^2 + ... + x^degree """
#Juste repris le truc de l'exo, mais peut être marche pas quand le nombre de feature > 1
def build_poly(X, degree):
    
    phi = np.zeros((len(X),degree+1))
    for i in range(len(X)):
        phi[i,:] = [X[i]**k for k in range(0,degree+1)]
    return phi