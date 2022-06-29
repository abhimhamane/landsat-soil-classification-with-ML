from pandas import read_csv
from pandas import DataFrame

from sklearn.neighbors import KNeighborsClassifier

from numpy import arange
from numpy import meshgrid
from numpy import array

def viz_mesh_grid():
    b1 = arange(start = 25, stop = 160, step=1)
    b2 = arange(start = 25, stop = 160, step=1)

    xx, yy = meshgrid(b1,b2)

    return xx, yy

def viz_cdf():
    xx, yy = viz_mesh_grid()
    cdf = DataFrame({'b1': xx.reshape(-1),
                    'b2': yy.reshape(-1)})

    cdf_arr = array(cdf)
    return cdf, cdf_arr