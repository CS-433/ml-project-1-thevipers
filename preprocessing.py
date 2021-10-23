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
    (by default set to 90%).
    If the percentage is below the threshold, missing values will be kept, but set to the mean of the column.
    Be careful : *missing value = -999*
    However, we think that for some features it makes no sense to arbitrarly set missing values to the mean as some features 
    are just meaningless for some samples (for instance : DER_deltaeta_jet_jet is meaningless for samples with 
    PRI_jet_num = 0). That is why we choose to set those missing values to 0 instead of the mean.
    """
    
    new_train = tx_train
    new_test = tx_test
    
    for col in range(tx_train.shape[1]) :
        
        feature = new_train[:, col]
        
        # we calculate what percentage of missing values there is in each column :
        miss_vals = np.sum(tx_train[:,col]==-999)/999
        percentage = miss_vals/tx_train[:,0].shape

        # we will change every missing value in the column to the mean of the column (arbitrary choice) :
        where = np.where(feature!=-999, True, False)
        mean = np.mean(feature, where=where)
        feature[feature==-999] = mean
        new_train[:,col] = feature

        # if the percentage of missing values in a column is above a treshold, we will just completely remove the column, 
        # because the column is considered useless for the prediction
        if(percentage >= threshold) :
            new_train = np.delete(new_train, col, axis=1)
            new_test = np.delete(new_test, col, axis=1)

    print(tx_train.shape[1]-new_train.shape[1], ' features have been removed')
            
    return new_train, new_test


def remove_useless_features(x_train, x_test) :
    '''
    This function removes the useless features of the parameter x :
        - We first remove the features 15, 16, 18 and 20, because we found out those features were useless to explain y
        according to the visualization plots in the section 'Features' distributions analysis'
        - Then we remove the features with a standard deviation of 0, because it means that the feature is useless to
        explain y
    '''
    
    # useless features found "by hand" using the vizualisation
    useless = [15, 16, 18, 20]
    new_x_train = np.delete(x_train, useless, axis=1)
    new_x_test = np.delete(x_test, useless, axis=1)
    
    print(x_train.shape[1]-new_x_train.shape[1], ' useless features have been removed')
    # std = 0
    std_x = np.std(new_x_train, axis=0)
    idx_std_0 = [i for i, std in enumerate(std_x) if std==0]
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


def outliers(x, y) :
    """
    Delete the outliers which are aberrant values, usually due to some errors in the experiment or from the material, 
    usually not relevant and may impact the prediction.
    """
    new_x=np.array(x)
    new_y=np.array(y)
    idx_outliers = []
    
    # find all the outliers
    for i in range(new_x.shape[1]) :
        idxs = findOutliers(x[:, i])
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



def findOutliers(x, outlierConstant=1.5):
    """
    Find the outliers using the Interquartile (IQR) method
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
    #if length ....
    N, D = X.shape 
    phi = np.zeros((N,(D*degree)+1))
    # intercept
    phi[:,0] = np.ones(N)
    # powers
    for deg in range(1,degree+1):
        for i in range(D):
            phi[:, 1+D*(deg-1)+i ] = np.power(X[:,i],deg)    
    return phi  

def process_data(x_train, x_test, alpha=0):
    """
    Preprocessing: impute missing values, feature engineering, delete outliers and standardization.
    """
    # Missing values:
    # consider the 0s in the 'PRI_jet_all_pt' as missing values
    x_train[:,-1]=np.where(x_train[:,-1]==0, -999, x_train[:,-1])
    # impute missing data
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

"""
def split_jet(train_x, train_y, tes_x, test_y) :
    
    
    return train_x_0, train_x_1, train_x_2, train_y_0, train_y_1, train_y_2, test_x_0, test_x_1, test_x_2, test_y_0, test_y_1, test_y_2,  
"""

#heavy tail -> log
#symmetric -> abs