#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import ndarray

from .baseClass import Operation


# In[ ]:



class Dropout(Operation):

    def __init__(self,
                 keep_prob: float = 0.8):
        super().__init__()
        self.keep_prob = keep_prob

    def _output(self) -> ndarray:
        self.mask = np.random.binomial(1, self.keep_prob,
                                           size=self.input_.shape)
        return self.input_ * self.mask

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * self.mask

