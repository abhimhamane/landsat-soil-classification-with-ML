from functools import cache
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

st.title("Soil Classification using ML")

col1, col2 = st.columns([3, 1])
col1.markdown("""
    This project is on understanding and developing intution for Machine Learning and Neural Networks. These 
    techniques/approaches cannot be and should not be treated as 'Black-Boxes'. Inthis project, I have tried to 
    show this fact visually by selecting a not much explored dataset of 'Soil Classification'. 
    
    First, we start with the dataset exploration and then develope some standard metrics such as spectral reflectance curve and
    pair-plots. I have tried different ML classifiers like KNN Classifier, Nearest Centroid, Random Forest, etc.
    The philosophy of these approaches can be visualized by their performance on the soil data. 
""")

col1.write("----------------------")

col1.header("About the Dataset")
col1.markdown("""
    1. This dataset was downloaded from the UCI Machine Learning Repository - https://archive.ics.uci.edu/ml/index.php
    2. Statlog (Landsat Satellite) Data Set - https://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29
""")

data_preview_container = col1.container()
data_preview_container.header("Dataset Preview")


raw_data_train = pd.read_csv("../Landsat_Soil_Data/Soil_test_dataset.csv", header=None)

print(raw_data_train)
data_preview_container.dataframe(raw_data_train)




col1.header("Contribute/Connect")
col1.markdown("""
    1. Github Repository for code - https://github.com/abhimhamane/soil_classification_ml
    2. For any suggestions and Feedback - connect @ https://www.linkedin.com/in/abhishekmhamane/
""")
