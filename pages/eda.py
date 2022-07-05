import streamlit as st
import pandas as pd
from soil_eda import *

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="",
    #layout="wide",
    initial_sidebar_state="expanded",

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
train_raw = load_raw_soil_data()
clean_data_train = load_cleaned_soil_data()
data_preview_container.dataframe(train_raw.head())


data = isolate_central_pxls(train_raw)

data_preview_container.header("Isolating Central Pixel Values")
data_head, data_tail = data_preview_container.columns([1,1])
data_head.subheader("Data Head")
data_head.dataframe(data.head())
data_tail.subheader("Data Tail")
data_tail.dataframe(data.tail())

data_stat_cont = data_preview_container.container()
#data_stat_choice = data_stat_cont.selectbox("Choose Stat to View:", options=['Spectral Reflectance Plot', 'Data count' ,'Data Box Plot', 'Data Scatter Plot', ])

data_preview_container.subheader("Spectral Reflectance Plot")
soil_dn, soil_std = grouped_stats(data)
spectral_plot = spectral_reflectance_plot(soil_dn, soil_std)
data_preview_container.pyplot(spectral_plot)

data_preview_container.subheader("Data count")
data_hist_plot = data_count_hist(data)
data_preview_container.pyplot(data_hist_plot)

data_preview_container.subheader("Data Box Plot - Train Data(Raw)")
train_box_plot = data_box_plot(data, "Train Box Plot")
data_preview_container.pyplot(train_box_plot)



data_preview_container.header("Cleaned Data")
clean_data_head, clean_data_tail =  data_preview_container.columns([1,1])

data_preview_container.subheader("Data Box Plot - Train Data(Cleaned)")
train_clean_plot = data_box_plot(clean_data_train, "Train Box Plot - Cleaned Data")
data_preview_container.pyplot(train_clean_plot)


data_preview_container.subheader("Data Scatter Plot")
data_preview_container.image('data_pairplot.png')





clean_data_head.subheader("Head of data")
clean_data_head.dataframe(clean_data_train.head())
clean_data_tail.subheader("Tail of data")
clean_data_tail.dataframe(clean_data_train.tail())




