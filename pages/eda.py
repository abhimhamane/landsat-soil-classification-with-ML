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
data_preview_container.markdown("""
For ease of data-handling only central pixels are used. Thus, reducing input feature size from 36 to just 4.
""")
data_head, data_tail = data_preview_container.columns([1,1])
data_head.subheader("Data Head")
data_head.dataframe(data.head())
data_tail.subheader("Data Tail")
data_tail.dataframe(data.tail())

data_stat_cont = data_preview_container.container()
#data_stat_choice = data_stat_cont.selectbox("Choose Stat to View:", options=['Spectral Reflectance Plot', 'Data count' ,'Data Box Plot', 'Data Scatter Plot', ])

data_stat_cont.subheader("Spectral Reflectance Plot")
data_stat_cont.markdown(
"""Using this data spectral reflectance curves were plotted. Spectral Reflectance curves are
unique for each feature, and are of vital importance in classification tasks""")
soil_dn, soil_std = grouped_stats(data)
spectral_plot = spectral_reflectance_plot(soil_dn, soil_std)
data_stat_cont.pyplot(spectral_plot)

data_stat_cont.subheader("Data count")
count_plot, count_desc = data_stat_cont.columns([1,1])
count_desc.markdown("""
The dataset has 6 soil classes. 

The frequency of each soil class is not same. Soil type 1 and 7
are over represented with 1072 and 1038 examples respectively. Soil type 3, 2, 5 and 4 have
959, 479, 470 and 415 examples respectively. 

Thus, the model is going to be biased towards
soil type 1 and 7.
""")

data_hist_plot = data_count_hist(data)
count_plot.pyplot(data_hist_plot)

data_stat_cont.subheader("Data Box Plot - Train Data(Raw)")
data_stat_cont.markdown(
"""The data could be visualized through the use of Box-plots and scatter plots. 

Box plots help in detecting the outliers, by presenting the 25th and 75th percentile along with the inter
quartile range. 
""")
train_box_plot = data_box_plot(data, "Train Box Plot")
data_stat_cont.pyplot(train_box_plot)


data_stat_cont.subheader("Data Scatter Plot")
data_stat_cont.markdown("""
Scatter plots help understand the presence of clustering, which often is the
case in such classification problems.

From Box-plot and pair plot it can be observed that there is natural clustering present in the
data but there are also many outliers. Outlier removal is required.

From the scatter plot it can also be observed that some band combinations are really helpful
such as – “b1 and b3” and “b1 and b2”. The cluster identification is much easier for these
band combinations.
""")
data_stat_cont.image('data_pairplot.png')



data_stat_cont.header("Cleaned Data")
data_stat_cont.markdown(
"""Data outliers can be identified if the data fall beyond the range b/w
Q1 – 1.5(Q3-Q1) and Q3 + 1.5(Q3-Q1), where Q1 and Q3 represent the first and third
quartiles respectively.

After removing the data outliers, cleaned data is obtained and this can be observed in the Scatter Pair Plot.


""")

clean_data_head, clean_data_tail =  data_stat_cont.columns([1,1])

data_stat_cont.subheader("Data Box Plot - Train Data(Cleaned)")
train_clean_plot = data_box_plot(clean_data_train, "Train Box Plot - Cleaned Data")
data_stat_cont.pyplot(train_clean_plot)





clean_data_head.subheader("Head of data")
clean_data_head.dataframe(clean_data_train.head())
clean_data_tail.subheader("Tail of data")
clean_data_tail.dataframe(clean_data_train.tail())




