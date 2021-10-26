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


def missing_values(tx_train, tx_test, threshold=0.9) :
    """
    Count the missing values of each feature and delete columns (features) above a certain threshold of missing values
    (by default set to 100%, i.e. columns containing only missing values).
    If the percentage is below the threshold, missing values will be kept, but set to the mean of the column.
    Be careful : *missing value = -999*
    """
    
    new_train = tx_train
    new_test = tx_test
    #Columns we will have to delete :
    cols = []
    
    for col in range(tx_train.shape[1]) :
        
        feature = new_train[:, col]
        
        # we calculate what percentage of missing values there is in each column :
        percentage =  np.count_nonzero(feature==-999)/len(feature)
        # we will change every missing value in the column to the meedian of the column (arbitrary choice) :
        median = np.median(feature[feature != -999])
        new_train[:,col] = np.where(feature==-999, median, new_train[:,col])
        new_test[:,col] = np.where(new_test[:,col]==-999, median, new_test[:,col])

        # if the percentage of missing values in a column is above a treshold, we will just completely remove the column, 
        # because the column is considered useless for the prediction
        if(percentage >= threshold) :
            cols.append(col)

    new_train = np.delete(new_train, cols, axis=1)
    new_test = np.delete(new_test, cols, axis=1)

    print(tx_train.shape[1]-new_train.shape[1], ' features have been removed')
    return new_train, new_test

####################
#Remove useless features and features with 0 standard deviation :

def remove_useless_features(x_train, x_test) :
    '''
    This function removes the useless features of the parameter x : We first remove the features 15, 16, 18 and 20, because we found out
    those features were useless to explain y according to the visualization plots in the section 'Features' distributions analysis'
    '''
    
    # useless features found "by hand" using the vizualisation
    useless = [15, 16, 18, 20]
    new_x_train = np.delete(x_train, useless, axis=1)
    new_x_test = np.delete(x_test, useless, axis=1)
    
    print(x_train.shape[1]-new_x_train.shape[1], ' useless features have been removed')

    return new_x_train , new_x_test

def remove_std_0(x_train, x_test) :
    '''
    This function removes the useless features of the parameter x : We remove the features with a standard deviation of 0, because it 
    means that the feature is useless to explain y
    '''
    
    new_x_train = x_train
    new_x_test = x_test
    
    # std = 0
    std_x = np.std(new_x_train, axis=0)
    idx_std_0 = np.argwhere(std_x == 0)
    new_x_train_ = np.delete(new_x_train, idx_std_0, axis=1)
    new_x_test_ = np.delete(new_x_test, idx_std_0, axis=1)
    print(new_x_train.shape[1]-new_x_train_.shape[1], ' features with a standard deviation equal to 0 have been removed')

    return new_x_train_ , new_x_test_



def standardize(x, mean_x=None, std_x=None):
    """
    Standardize the dataset.
    """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    if std_x is None:
        std_x = np.std(x, axis=0)
    
    x_std = x - mean_x
    x_std = x_std[:, std_x > 0] / std_x[std_x > 0]
    
    std_0 = std_x[std_x <=0]
    if(len(std_0)>0) :
        raise ValueError("DIVISION BY 0 : There are features with standard deviation equal to 0")
    
    return x_std, mean_x, std_x



def correlation(tx_train, tx_test, treshold=0.99) :
    """
    May be useful if we want to delete very correlated features
    """
    
    new_train = tx_train
    new_test = tx_test
    
    correlation_matrix = np.corrcoef(tx_train)
    corr_features = []
    
    for i in range(correlation_matrix.shape[0]):
        for j in range(i):
            if abs(correlation_matrix[i,j]) > treshold :
                corr_features.append(i)
    
    new_train = np.delete(new_train, corr_features, axis=1)
    new_test = np.delete(new_test, corr_features, axis=1)
    
    return new_train, new_test


####################
#Outliers :

def outliers(x, y, alpha=5) :
    """
    Delete the outliers which are aberrant values, usually due to some errors in the experiment or from the material, 
    usually not relevant and may impact the prediction.
    """
    new_x=np.array(x)
    new_y=np.array(y)
    idx_outliers = []
    
    # find all the outliers
    for i in range(new_x.shape[1]) :
        idxs = findOutliers(x[:, i], alpha=alpha)
        for j, idx in enumerate(idxs) :
            idx_outliers.append(idx)

    # delete the lines containing outliers
    if(len(idx_outliers)>0) :
        idx_outliers = np.array(list(set(idx_outliers)))
        new_x = np.delete(new_x, idx_outliers, axis=0)
        new_y = np.delete(new_y, idx_outliers, axis=0)
        print('With outliers : ', x.shape)
        print('Without outliers : ', new_x.shape)
        return new_x, new_y
    else :
        print('There are no significative outliers')
        return new_x, new_y


def findOutliers(x, alpha, outlierConstant=1.5):
    """
    Find the outliers using the Interquartile (IQR) method
    Return the indices of the outliers
    """
    a = np.array(x)
    upper_quartile = np.percentile(a, 100-alpha)
    lower_quartile = np.percentile(a, alpha)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    outliers_idx = []
    for i, y in enumerate(a.tolist()):
        if y <= quartileSet[0] or y >= quartileSet[1]:
            outliers_idx.append(i)
    return outliers_idx


#####################

def build_poly(X, degree):
    """
    Feature engineering by polynomial expansion: add an intercept and for each feature,
    add a polynomial expansion from 1 to degree.
    """
    #if length ....
    N, D = X.shape 
    phi = np.zeros((N,(D*degree)+1))
    # intercept
    phi[:,0] = np.ones(N)
    # powers
    for deg in range(1,degree+1):
        for i in range(D):
            phi[:, D*(deg-1)+i ] = np.power(X[:,i],deg)    
    return phi  



"""
def split_jet(train_x, train_y, tes_x, test_y) :
    
    
    return train_x_0, train_x_1, train_x_2, train_y_0, train_y_1, train_y_2, test_x_0, test_x_1, test_x_2, test_y_0, test_y_1, test_y_2,  
"""

#heavy tail -> log
#symmetric -> abs