# -*- coding: utf-8 -*-
"""
GENERAL DESCRIPTION :
The pre-processing aims to modify the raw data into an easily analyzed data set
For instance we can delete the 'useless' features (column of zeros, correlated values,...), standarize,  deal with missing values, NaN, ...
"""
import numpy as np


def offset(x) :
    """ 
    This function aims to add a first column of 1 in X (usually called the offset).
    """
    new_x = np.c_[np.ones(x.shape[0]), x]
    return new_x


#QUESTION : Est ce que les 0 sont aussi des missing values ????
# Seulement dans la colonne PRI_jet_all_pt ??? J'AI POSÉ LA QUESTION !!
def missing_values(tx_train, tx_test, threshold=0.9) :
    """
    Count the missing values of each feature and delete columns (features) above a certain threshold of missing values
    (by default set to 90%).
    If the percentage is below the threshold, missing values will be kept, but set to the mean of the column.
    Be careful : *missing value = -999*
    """
    
    new_train = tx_train
    new_test = tx_test
    
    for col in range(tx_train.shape[1]) :
        
        # We calculate what percentage of missing values there is in each column :
        miss_vals = np.sum(tx_train[:,col]==-999)/999
        percentage = miss_vals/tx_train[:,0].shape
        
        # We will change every missing value in the column to the mean of the column (arbitrary choice) :
        feature = new_train[:, col]
        where = np.where(feature!=-999, True, False)
        mean = np.mean(feature, where=where)
        feature[feature==-999] = mean
        new_train[:,col] = feature
        
        # If the percentage of missing values in a column is above a treshold, we will just completely remove the column, 
        # because the column is considered useless for the prediction
        if(percentage >= threshold) :
            new_train = np.delete(new_train, col, axis=1)
            new_test = np.delete(new_test, col, axis=1)
            
    return new_train, new_test


def standardize(x, mean_x=None, std_x=None):
    """
    Standardize the dataset.
    """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    if std_x is None:
        std_x = np.std(x, axis=0)
    
    x_std = (x - mean_x)/std_x
    
    return x_std, mean_x, std_x


#???? Do we need it ???? ÇA PEUT ÊTRE BIEN D'ENLEVER 1 DES 2 FEATURES SI IL Y EN A 2 QUI SONT CORRÉLÉES À 99% !
def correlation() :
    
    return 0


def outliers(x) :
    """
    MAIS DU COUP LÀ TU ENLÈVES LA COLONNE QUI CONTIENT UN OUTLIER ?!
    IL FAUDRAIT PLUTÔT ENLEVER LE DIT SAMPLE PLUTÔT QUE LA FEATURE…
    Delete the outliers which are aberrant values, usually due to some errors in the experiment or from the material, 
    usually not relevant and may impact the prediction.
    """
    new_x=x
    for i in range(x.shape[1]) :
        idx = findOutliers(x[:, i])
        np.delete(new_x, idx, axis=0)
    return new_x
    

def findOutliers(x, outlierConstant=1.5):
    """
    À VÉRIFIER SI LÀ ON EST EN TRAIN DE COMPARER BIEN LES OUTLIERS COLONNE PAR COLONNE 
    COMMENTAIRE PAS CORRECT
    Cut the tails: if a value is smaller than alpha_percentile (bigger than 1-alpha_percentile) 
    of its features replace it with that percentile
    Return the indices of the outliers
    """
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


def build_poly(X, degree):
    """
    Feature engineering by polynomial expansion: add an intercept and for each feature,
    add a polynomial expansion from 1 to degree.
    """
    N, D = X.shape 
    phi = np.zeros((N,(D*degree)+1))
    # intercept
    phi[:,0] = np.ones(N)
    # powers
    for deg in range(1,degree+1):
        for i in range(D):
            poly[:, 1+D*(deg-1)+i ] = np.power(X[:,i],deg)    
    return phi  

def process_data(x_train, x_test, alpha=0):
    """
    Preprocessing: impute missing values, feature engineering, delete outliers and standardization.
    """
    # Missing Values:
    # Consider the 0s in the 'PRI_jet_all_pt' as missing values
    x_train[:,-1]=np.where(x_train[:,-1]==0, -999, x_train[:,-1])
    # Impute missing data
    x_train, x_test = missing_values(x_train, x_test) 
    
    # Feature Engineering:
    # Absolute value of symmetrical features
  
    # Other transformation for positive features

    # Delete useless features

    
    # Delete outliers
    x_train = outliers(x_train)
    x_test = outliers(x_test)
    
    # Standardization
    x_train, mean_x_train, std_x_train = standardize(x_train)
    x_test, _, _ = standardize(x_test, mean_x_train, std_x_train)
     
    return x_train, x_test

