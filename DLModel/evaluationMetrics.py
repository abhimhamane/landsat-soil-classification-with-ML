#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy import ndarray
from typing import List

from .network import NeuralNetwork
from .utility.np_utility import softmax



def mae(y_true: ndarray, y_pred: ndarray):
    '''
    Compute mean absolute error for a neural network.
    '''    
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: ndarray, y_pred: ndarray):
    '''
    Compute root mean squared error for a neural network.
    '''
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))
   

def eval_regression_model(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray):
    '''
    Compute mae and rmse for a neural network.
    '''
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print()
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))


def model_prediction(model: NeuralNetwork,
                          X: ndarray):
    
    preds_ = model.forward(X)
    return preds_

def classification_accuracy(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray, no_class: int = 6):
    '''
    Compute basic classification accuracy for a neural network.
    '''
    preds = model.forward(X_test)
    #preds = softmax(X_test)
    
    lbl = []
    for i in range(len(preds)):
        lbl_temp = np.zeros((no_class))
        max_id = max_index(preds[i])
        lbl_temp[max_id] = 1

        lbl.append(lbl_temp)

    preds = lbl

    counts = 0
    samples = len(y_test)

    idx = np.where(y_test==1)
    for i in range(len(y_test)):
        b = idx[1]
        b = b[i]

        if y_test[i][b] == preds[i][b]:
            counts += 1
            
    return ((counts/samples)*100)

def get_clf_pred(model: NeuralNetwork,
                          X_test: ndarray,no_class: int = 6):
    preds = model.forward(X_test)
    preds = softmax(X_test)
    
    lbl = []
    for i in range(len(preds)):
        lbl_temp = np.zeros((no_class))
        max_id = max_index(preds[i])
        lbl_temp[max_id] = 1

        lbl.append(lbl_temp)

    preds = lbl
    return preds

def max_index(x_):
    x_ = np.array(x_)
    for i in range(len(x_)):
        max_ = np.max(x_)
        if x_[i] == max_:
            return i


