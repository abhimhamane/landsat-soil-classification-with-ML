import streamlit as st
import pandas as pd
from soil_eda import *

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="",
)

st.title("Exploratory Data Analysis")
st.markdown("""
    The database consists of the multi-spectral values of pixels in 3x3 neighborhood's 
    in a satellite image, and the classification is associated with the central pixel in each 
    neighbourhood. 

    The aim is to predict this classification, given the multi-spectral values. In the sample database, 
    the class of a pixel is coded as a number.
""")

data_preview_container = st.container()
data_preview_container.header("Raw Data Preview")
train_raw, test_raw = load_raw_soil_data()
data_preview_container.dataframe(train_raw.head())


data = isolate_central_pxls(train_raw)

data_preview_container.header("Isolating Central Pixel Values")
data_head, data_tail = data_preview_container.columns([1,1])
data_head.subheader("Data Head")
data_head.dataframe(data.head())
data_tail.subheader("Data Tail")
data_tail.dataframe(data.tail())

data_stat_cont = data_preview_container.container()
data_stat_cont.write("Value counts:")
data_stat_cont.write(data['soil_type'].value_counts())


data_preview_container.header("Cleaned Data")
clean_data_head, clean_data_tail =  data_preview_container.columns([1,1])

clean_data_train = load_cleaned_soil_data()

clean_data_head.subheader("Head of data")
clean_data_head.dataframe(clean_data_train.head())
clean_data_tail.subheader("Tail of data")
clean_data_tail.dataframe(clean_data_train.tail())




