import streamlit as st

from pandas import read_csv
from pandas import DataFrame
from numpy import arange
from numpy import meshgrid
from numpy import array

st.set_page_config(
    page_title="Classifier Models",
    page_icon="",
)

st.title("Classifier Models")

# Custom Functions
def viz_mesh_grid():
    b1 = arange(start = 25, stop = 160, step=1)
    b2 = arange(start = 25, stop = 160, step=1)

    xx, yy = meshgrid(b1,b2)

    cdf = DataFrame({'b1': xx.reshape(-1),
                    'b2': yy.reshape(-1)})

    cdf_arr = array(cdf)
    return cdf_arr

# import clean dataset
data = read_csv(r'./Clean_Data/clean_trainData.csv')
test = read_csv(r'./Clean_Data/clean_testData.csv')

print(data)
# Sidebar
sidebar = st.sidebar
sidebar.header("")

# selecting bands for model training 

#train_bands = array(data[[band_1, band_2]])
#test_bands = array(test[[band_1, band_2]])

train_yy = array(data[['soil_type']])
test_yy = array(test[['soil_type']])