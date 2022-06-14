#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import ndarray

from .baseClass import ParamOperation


# In[2]:


"""
class WeightMultiply(ParamOperation):

    def __init__(self, W: ndarray):
        super().__init__(W)

    def _output(self) -> ndarray:
        return np.matmul(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return np.matmul(output_grad, self.param.transpose(1, 0))

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        return np.matmul(self.input_.transpose(1, 0), output_grad)
"""

class WeightMultiply(ParamOperation):
    '''
    Weight multiplication operation for a neural network.
    '''

    def __init__(self, W: ndarray):
        '''
        Initialize Operation with self.param = W.
        '''
        super().__init__(W)

    def _output(self) -> ndarray:
        '''
        Compute output.
        '''
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        '''
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad: ndarray)  -> ndarray:
        '''
        Compute parameter gradient.
        '''        
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


# In[3]:


class BiasAdd(ParamOperation):
    '''
    Compute bias addition.
    '''

    def __init__(self,
                 B: ndarray):
        '''
        Initialize Operation with self.param = B.
        Check appropriate shape.
        '''
        assert B.shape[0] == 1
        
        super().__init__(B)

    def _output(self) -> ndarray:
        '''
        Compute output.
        '''
        return self.input_ + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        '''
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute parameter gradient.
        '''
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])

