#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import ndarray

from .utility.np_utility import (assert_same_shape,
                                 softmax,
                                 normalize,
                                #exp_ratios,
                                unnormalize)


# In[2]:


class Loss(object):
    '''
    The "loss" of a neural network
    '''

    def __init__(self):
        '''Pass'''
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        '''
        Computes the actual loss value
        '''
        #print(prediction.shape," loss ",target.shape)
        
        assert_same_shape(prediction, target)

        self.prediction = prediction
        self.target = target

        loss_value = self._output()

        return loss_value

    def backward(self) -> ndarray:
        '''
        Computes gradient of the loss value with respect to the 
        input to the loss function
        '''
        self.input_grad = self._input_grad()

        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self) -> float:
        '''
        Every subclass of "Loss" must implement the _output function.
        '''
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        '''
        Every subclass of "Loss" must implement the _input_grad 
        function.
        '''
        raise NotImplementedError()


# In[ ]:


class MeanSquaredError(Loss):

    def __init__(self,
                 normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize

    def _output(self) -> float:

        if self.normalize:
            self.prediction = self.prediction / self.prediction.sum(axis=1, keepdims=True)

        loss = np.sum(np.power(self.prediction - self.target, 2)) / self.prediction.shape[0]

        return loss

    def _input_grad(self) -> ndarray:

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


# In[3]:


class MeanSquaredError_old(Loss):

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> float:
        '''
        Computes the per-observation squared error loss
        '''
        loss = (
            np.sum(np.power(self.prediction - self.target, 2)) / 
            self.prediction.shape[0]
        )

        return loss

    def _input_grad(self) -> ndarray:
        '''
        Computes the loss gradient with respect to the input for MSE loss
        '''        

        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


# In[ ]:


class SoftmaxCrossEntropy(Loss):
    def __init__(self, eps: float=1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.single_class = False

    def _output(self) -> float:

        # if the network is just outputting probabilities
        # of just belonging to one class:
        if self.target.shape[1] == 0:
            self.single_class = True

        # if "single_class", apply the "normalize" operation defined above:
        if self.single_class:
            self.prediction, self.target =             normalize(self.prediction), normalize(self.target)

        # applying the softmax function to each row (observation)
        softmax_preds = softmax(self.prediction, axis=1)

        # clipping the softmax output to prevent numeric instability
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

        # actual loss computation
        softmax_cross_entropy_loss = (
            -1.0 * self.target * np.log(self.softmax_preds) - \
                (1.0 - self.target) * np.log(1 - self.softmax_preds)
        )

        return np.sum(softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:

        # if "single_class", "un-normalize" probabilities before returning gradient:
        if self.single_class:
            return unnormalize(self.softmax_preds - self.target)
        else:
            return (self.softmax_preds - self.target) / self.prediction.shape[0]


# In[ ]:


class SoftmaxCrossEntropyComplex(SoftmaxCrossEntropy):
    def __init__(self, eta: float=1e-9,
                 single_output: bool = False) -> None:
        super().__init__()
        self.single_output = single_output

    def _input_grad(self) -> ndarray:

        prob_grads = []
        batch_size = self.softmax_preds.shape[0]
        num_features = self.softmax_preds.shape[1]
        for n in range(batch_size):
            exp_ratio = exp_ratios(self.prediction[n] - np.max(self.prediction[n]))
            jacobian = np.zeros((num_features, num_features))
            for f1 in range(num_features):  # p index
                for f2 in range(num_features):  # SCE index
                    if f1 == f2:
                        jacobian[f1][f2] = (
                            self.softmax_preds[n][f1] - self.target[n][f1])
                    else:
                        jacobian[f1][f2] = (
                            -(self.target[n][f2]-1) * exp_ratio[f1][f2] + self.target[n][f2] + self.softmax_preds[n][f1] - 1)
            prob_grads.append(jacobian.sum(axis=1))

        if self.single_class:
            return unnormalize(np.stack(prob_grads))
        else:
            return np.stack(prob_grads)

