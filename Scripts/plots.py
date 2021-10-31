# -*- coding: utf-8 -*-
"""Visualisation methods."""
import numpy as np
import matplotlib.pyplot as plt



def CV_param_plot(params, acc_tr, acc_te, name_param="degree") :
    """
    Visualise the curves of the training accuracy and test accuracy according to a parameter we want to tune.
    Takes as input:
        * the different values of the parameter to tune : params
        * the associated training accuracies : acc_tr
        * the associated test accuracies : acc_te
        * the name of the parameter to tune : name_param
    """
    if(name_param=="degree") :
        plt.plot(params, acc_tr, marker=".", color='b', label='train accuracy')
        plt.plot(params, acc_te, marker=".", color='r', label='test accuracy')
    else :
        plt.semilogx(params, acc_tr, marker=".", color='b', label='train accuracy')
        plt.semilogx(params, acc_te, marker=".", color='r', label='test accuracy')
        
    plt.xlabel(name_param)
    plt.ylabel("Accuracy")
    plt.ylim(np.min(acc_te)-0.01,np.max(acc_tr)+0.01)
    plt.title("Accuracy in function of the "+name_param+" using cross-validation")
    plt.legend(loc=2)
    plt.grid(True)
    #plt.savefig(name_param+"_CV")
    plt.show()
    
def compare(methods, accuracies) :
    """
    Compare and visualise the test accuracies of the different methods we have used.
    Takes as input:
        * the name of the methods we want to compare : methods
        * the associated test accuracies : accuracies
    """
    
    plt.figure(figsize=(10,6)) 
    plt.plot(methods, accuracies, 'bo')
    
    plt.xticks(rotation=45, fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.ylim(0,1)
    plt.title("Accuracy of each method using its best tuned parameter", fontsize=15)
    plt.grid(True)
    plt.savefig("Comparison of methods", dpi=300, bbox_inches='tight')
    plt.show()
    