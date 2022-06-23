import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="",
)

st.title("Exploratory Data Analysis")
st.markdown("""

""")

data_preview_container = st.container()
data_preview_container.header("Raw Data Preview")
raw_data_train = pd.read_csv("./Landsat_Soil_Data/Soil_train_dataset.csv", header=None)
data_preview_container.dataframe(raw_data_train.head())

data_preview_container.header("Clean Data")
clean_data_head, clean_data_tail =  data_preview_container.columns([1,1])

clean_data_train = pd.read_csv("./Clean_Data/clean_trainData.csv")

clean_data_head.subheader("Head of data")
clean_data_head.dataframe(clean_data_train.head())
clean_data_tail.subheader("Tail of data")
clean_data_tail.dataframe(clean_data_train.tail())




