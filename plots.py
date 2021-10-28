# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt



def CV_param_plot(params, acc_tr, acc_te, name_param="degree"):
    """visualization the curves of mse_tr and mse_te in function of a tuned parameter"""
    if(name_param=="degree") :
        plt.plot(params, acc_tr, marker=".", color='b', label='train accuracy')
        plt.plot(params, acc_te, marker=".", color='r', label='test accuracy')
    else :
        plt.semilogx(params, acc_tr, marker=".", color='b', label='train accuracy')
        plt.semilogx(params, acc_te, marker=".", color='r', label='test accuracy')
        
    plt.xlabel(name_param)
    plt.ylabel("Accuracy")
    plt.title("Accuracy in function of the "+name_param+" using cross-validation")
    plt.legend(loc=2)
    plt.grid(True)
    #plt.savefig(name_param+"_CV")
    plt.show()
    
def compare(methods, accuracies) :
    
    plt.plot(methods, accuracies, 'bo')
    
    plt.xticks(rotation=90)
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of each method using its best tuned parameter")
    plt.grid(True)
    #plt.savefig("Comparison of methods")
    plt.show()
    