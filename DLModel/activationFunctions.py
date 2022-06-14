#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import ndarray


from DLModel.baseClass import Operation
from .utility.np_utility import softmax

# In[ ]:


class Sigmoid(Operation):
    '''
    Sigmoid activation function.
    '''

    def __init__(self) -> None:
        '''Pass'''
        super().__init__()

    def _output(self) -> ndarray:
        '''
        Compute output.
        '''
        return 1.0/(1.0+np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Compute input gradient.
        '''
        sigmoid_backward = self.output * (1.0 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


# In[ ]:


class Linear(Operation):
    '''
    "Identity" activation function
    '''

    def __init__(self) -> None:
        '''Pass'''        
        super().__init__()

    def _output(self) -> ndarray:
        '''Pass through'''
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Pass through'''
        return output_grad


# In[ ]:


class Tanh(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

   # def _output(self, inference: bool) -> ndarray:
    def _output(self) -> ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        return output_grad * (1 - self.output * self.output)


# In[ ]:


class ReLU(Operation):
    '''
    Hyperbolic tangent activation function
    '''
    def __init__(self) -> None:
        super().__init__()

    #def _output(self, inference: bool) -> ndarray:
    def _output(self) -> ndarray:
        return np.clip(self.input_, 0, None)

    def _input_grad(self, output_grad: ndarray) -> ndarray:

        mask = self.output >= 0
        #print(output_grad * mask)
        return output_grad * mask


# In[2]:


# Implementation of SeLU
# _Self-Normalizing Neural Networks: https://arxiv.org/abs/1706.02515
class SELU(Operation):
    '''
    Scaled Exponential Linear Unit
    '''
    
    def __init__(self) -> None:
        super().__init__()
        self._scale = 1.0507009873554804934193349852946
        self._alpha = 1.6732632423543772848170429916717

    def _output(self) -> ndarray:
        
        # masks the -ve elements
        _mask_pos = self.input_ >= 0
        # if X >=0 -> selu(x) = scale*x
        _x1_ = _mask_pos * self._scale * self.input_
        
        # masks the +ve elements
        _mask_neg = self.input_ < 0
        # if x < 0 -> selu(x) = scale * alpha * (exp(x)-1)
        _x2_ = _mask_neg * self._alpha * self._scale * (np.exp(self.input_) - 1.0)
        
        # adding both terms to get final result
        _opt_ = _x1_ + _x2_
        #print(_opt_)
        return _opt_
        
    def _input_grad(self, output_grad:ndarray) -> ndarray:
        
        # masks the -ve elements
        _mask_neg = self.input_ >= 0
        # if x >=0 -> f'(x) = scale
        _x1 = self._scale * _mask_neg
        
        # masks the +ve elements
        _mask_pos = self.input_ < 0
        # if x < 0 -> f'(x) = alpha * scale * exp(x)
        _x2 = self._scale * self._alpha * np.exp(self.input_) * _mask_pos
        
        # adding both terms to get final result
        _opt = _x1 + _x2
        #print(_opt)
        return output_grad * _opt

class Swish(Operation):
    '''
    Swish Activation Function
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def _output(self) -> ndarray:
        _sig = 1.0/(1.0+np.exp(-1.0 * self.input_))
        return self.input_ * _sig
    
    def _input_grad(self, output_grad:ndarray) -> ndarray:
        _opt1 = self.input_ * (Sigmoid(self.input_) * (1 - Sigmoid(self.input_))) + Sigmoid(self.input_)
        return _opt1 * output_grad

class Softmax(Operation):
    '''
    SoftMax Activation Function
    '''
    def __init__(self) -> None:
        super().__init__()
    def _output(self) -> ndarray:
        _softmax = softmax(self.input_)
        return _softmax
    def _input_grad(self, output_grad:ndarray) -> ndarray:
        pass


