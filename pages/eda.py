import streamlit as st
import pandas as pd

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
train_df = pd.read_csv("./Landsat_Soil_Data/Soil_train_dataset.csv", header=None)
data_preview_container.dataframe(train_df.head())




train_X = train_df.loc[:, 16:19]
train_Y = train_df.loc[:, 36]

data = pd.concat([train_X, train_Y], axis=1)
data.columns = ['b1', 'b2', 'b3', 'b4', 'soil_type']


data_preview_container.dataframe(data.head())









data_preview_container.header("Clean Data")
clean_data_head, clean_data_tail =  data_preview_container.columns([1,1])

clean_data_train = pd.read_csv("./Clean_Data/clean_trainData.csv")

clean_data_head.subheader("Head of data")
clean_data_head.dataframe(clean_data_train.head())
clean_data_tail.subheader("Tail of data")
clean_data_tail.dataframe(clean_data_train.tail())




