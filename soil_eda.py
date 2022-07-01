import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sr

def load_raw_soil_data():
    """
    Returns soil dataset as Pandas dataframe
    """
    train_df = pd.read_csv("Landsat_Soil_Data/Soil_train_dataset.csv", header=None)
    test_df = pd.read_csv("Landsat_Soil_Data/Soil_test_dataset.csv", header=None)
    return train_df, test_df

def load_cleaned_soil_data():
    """
    Returns cleaned soil dataset as Pandas dataframe
    """
    clean_train_df = pd.read_csv("Clean_Data/clean_trainData.csv")
    clean_test_df = pd.read_csv("Clean_Data/clean_testData.csv")
    return clean_train_df

def isolate_central_pxls(dataset):
    """
    Isolates the central 4 pxiels from the raw dataset and 
    returns a pandas dataframe with header
    """
    data_x = dataset.loc[:, 16:19]
    data_y = dataset.loc[:, 36]

    data = pd.concat([data_x, data_y], axis=1)
    data.columns = ['b1', 'b2', 'b3', 'b4', 'soil_type']

    return data

def grouped_stats(data):
    soil_grp = data.groupby(['soil_type'])
    grp_stats = soil_grp.describe()

    soil_dn = []
    soil_std = []

    bands = ['b1', 'b2', 'b3', 'b4']
    soil_types = [0, 1, 2, 3, 4, 5]
    
    
    for soil in soil_types:
        soil_ind = []
        soil_ind_std = []
        for band in bands:
            soil_ind.append(grp_stats[band].iloc[soil]['mean'])
            soil_ind_std.append(grp_stats[band].iloc[soil]['std'])
        
        soil_dn.append(soil_ind)
        soil_std.append(soil_ind_std)
    
    return soil_dn, soil_std

def spectral_reflectance_plot(soil_dn, soil_std):

    bands = ['b1', 'b2', 'b3', 'b4']

    fig = plt.figure()
    plt.xlabel("Bands")
    plt.ylabel("DN Number")
    plt.title("Spectral Reflectance Curve")
    plt.errorbar(bands, soil_dn[0] , soil_std[0], label='soil_1')
    plt.errorbar(bands, soil_dn[1] , soil_std[1], label='soil_2')
    plt.errorbar(bands, soil_dn[2] , soil_std[2], label='soil_3')
    plt.errorbar(bands, soil_dn[3] , soil_std[3], label='soil_4')
    plt.errorbar(bands, soil_dn[4] , soil_std[4], label='soil_5')
    plt.errorbar(bands, soil_dn[5] , soil_std[5], label='soil_7')
    plt.legend(loc='best', ncol=2)
    return plt

def data_box_plot(data):
    
    plt.subplot(2,2,1)
    sr.boxplot(x = 'soil_type', y = 'b1', data=data)

    plt.subplot(2,2,2)
    sr.boxplot(x = 'soil_type', y = 'b2', data=data)

    plt.subplot(2,2,3)
    sr.boxplot(x = 'soil_type', y = 'b3', data=data)

    plt.subplot(2,2,4)
    sr.boxplot(x = 'soil_type', y = 'b4', data=data)

    return plt

def data_pair_plot(data):
    pg = sr.PairGrid(data, hue='soil_type', diag_sharey=True,palette="deep", height=3.0 , aspect=1.2)
    #pg.map(sr.scatterplot)
    #pg.map_offdiag(sr.scatterplot)
    pg.map_diag(sr.histplot, )
    pg.map_lower(sr.scatterplot, alpha=0.2)
    pg.map_upper(sr.kdeplot)
    #pg.map_diag(sr.boxplot)
    pg.add_legend()
    pg.fig.suptitle("Your Title", y=1.08)
    pg

def bound(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    median = df.median()
    std_dev = df.std()
    
    lower_bound = Q1 - (1.0*IQR)
    upper_bound = Q3 + (1.0*IQR)
    
    #lower_bound = median - (1*std_dev)
    #upper_bound = median + (1*std_dev)
    
    return(lower_bound, upper_bound)

def outlier_removal(data):
    a,b = data.shape
    ls = []
    for i in range(a):
        type = data.iloc[i,4]
        b1 = data.iloc[i,0]
        b2 = data.iloc[i,1]
        b3 = data.iloc[i,2]
        b4 = data.iloc[i,3]
        
        l1, u1 = bound(data.groupby(['soil_type']).get_group(type)['b1'])
        l2, u2 = bound(data.groupby(['soil_type']).get_group(type)['b2'])
        l3, u3 = bound(data.groupby(['soil_type']).get_group(type)['b3'])
        l4, u4 = bound(data.groupby(['soil_type']).get_group(type)['b4'])
        
        if(b1<l1 or b2<l2 or b3 < l3 or b4<l4 or b1>u1 or b2>u2 or b3>u3 or b4>u4):
            ls.append(i)
    
    cleaned_data = data.drop(ls)
    #cleaned_data = pd.DataFrame(cleaned_data)

    return cleaned_data



"""
# Testing the functions
data = isolate_central_pxls(load_raw_soil_data())
#print(grouped_stats(data)[1])

pp = spectral_reflectance_plot(grouped_stats(data)[0], grouped_stats(data)[1])

#box = data_box_plot(data)

#pp_plot = data_pair_plot(data)
clean_data = outlier_removal(data)
box2 = data_box_plot(clean_data)
box2.show()
#print(clean_data.shape, data.shape)
"""
