from decimal import MIN_EMIN
from pyexpat import model
from xml.sax.handler import all_properties
from click import option
from requests import options
import streamlit as st

from pandas import read_csv
from pandas import DataFrame

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from numpy import arange
from numpy import meshgrid
from numpy import array

import seaborn as sns
from matplotlib import pyplot as plt

from ml_models import KNN_info

st.set_page_config(
    page_title="Classifier Models",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",

)

st.title("Classifier Models")

# Custom Functions
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

# import clean dataset
data = read_csv(r'./Clean_Data/clean_trainData.csv')
test = read_csv(r'./Clean_Data/clean_testData.csv')

# Sidebar
st.sidebar.header("Parms")

model_select = st.sidebar.selectbox(label="Select ML Classifier:", options=["KNN Classifier", "Nearest Centroid", "Decision Trees", "Random Forest", "Neural Network"])

band_select = st.sidebar.form("select_band-(Features)")
band_1 = band_select.selectbox("Select Band 1-(Feature 1):", options=['b1', 'b2', 'b3', 'b4'], index = 0)
band_2 = band_select.selectbox("Select Band 2-(Feature 2):", options=['b1', 'b2', 'b3', 'b4'], index = 1)
band_select.form_submit_button("Apply Bands")



# selecting bands for model training 

train_bands = array(data[[band_1, band_2]])
test_bands = array(test[[band_1, band_2]])

train_yy = array(data[['soil_type']])
test_yy = array(test[['soil_type']])

model_prams_plots = st.container()

if model_select == "KNN Classifier":
    model_params, train_plot = model_prams_plots.columns([1,1])
    model_params.subheader(model_select)
    #knn_model_descri, knn_model_cont = model_params.columns([2, 1])

    knn_params_form = model_params.form("KNN Params Form")
    knn_params_form.text("KNN Params Form")
    n_ngbrs = knn_params_form.slider("N Nearest Neighbours:", min_value=1, max_value=50, value=5, step=1)
    weights = knn_params_form.selectbox("Weight Function:" ,options=["uniform", "distance"], index=0)
    algo = knn_params_form.selectbox("Nearest Neighbour Algorithm:", options=['auto', 'ball_tree', 'kd_tree', 'brute'], index=0)
    knn_params_form.form_submit_button("Apply Params")

    #train_plot, test_plot = st.columns([1,1])
    
    # initiate Model classifier
    knn = KNeighborsClassifier(n_neighbors=n_ngbrs, weights=weights, algorithm=algo)
    knn_hyperparameters = ['n_neighbors', 'weights', 'algorithm']
    knn.fit(train_bands, train_yy.ravel())

    # Model Params
   # model_params.write(knn.get_params())


    # Vizualization
    _xx, _yy = viz_mesh_grid()
    cdf, cdf_arr = viz_cdf()
    knnz = knn.predict(cdf_arr)
    cdf['predict'] = knnz
    
    
    train_fig, train_knn_plot = plt.subplots()
    train_knn_plot.contourf(_xx, _yy, knnz.reshape(_xx.shape))
    sns.scatterplot(x=data[band_1], y=data[band_2], hue=data.soil_type, palette="bright", marker='+')
    plt.title("KKN Visualization (Training Data)")
    train_plot.pyplot(train_fig)

    # Model Train Score
    train_plot.write(knn.score(train_bands, train_yy, sample_weight=None))

    
   
    
    

    # Vizualization of trained model on test data 
    #test_fig, test_knn_plot = plt.subplots()
    #test_knn_plot.contourf(_xx, _yy, knnz.reshape(_xx.shape))
    #sns.scatterplot(x=test[band_1], y=test[band_2], hue=test.soil_type, palette="bright", marker='+')
    #plt.title("KKN Visualization (Testing Data)")
    #test_plot.pyplot(test_fig)

    # Model Test Score
    train_plot.write(knn.score(test_bands, test_yy, sample_weight=None))
  


elif model_select == "Nearest Centroid":
    model_params, train_plot = model_prams_plots.columns([1,1])
    model_params.subheader(model_select)
    #nearest_centroid_descri, nearest_centroid_cont = model_params.columns([2, 1])
    NC_params_form = model_params.form("NC Params Form")
    NC_params_form.text("Nearest Centroid Params Form")
    metric = NC_params_form.selectbox("Metrics:", options=['euclidean', 'manhattan','cityblock', 'cosine', 'l1', 'l2'], index=0)
    NC_params_form.form_submit_button("Apply Params")

    nc = NearestCentroid(metric=metric)
    nc.fit(train_bands, train_yy.ravel())

    # Model Params
    #st.write(nc.get_params())

    # Vizualization of trained model on trained data 
    _xx, _yy = viz_mesh_grid()
    cdf, cdf_arr = viz_cdf()
    ncz = nc.predict(cdf_arr)
    cdf['predict'] = ncz
    
    #train_plot, test_plot = st.columns([1,1])
    train_fig, train_nc_plot = plt.subplots()
    train_nc_plot.contourf(_xx, _yy, ncz.reshape(_xx.shape))
    sns.scatterplot(x=data[band_1], y=data[band_2], hue=data.soil_type, palette="bright", marker='+')
    plt.title("Nearest Centroid Visualization (Training Data)")
    train_plot.pyplot(train_fig)
    # Model Train Score
    train_plot.write(nc.score(train_bands, train_yy, sample_weight=None))

    # Vizualization of trained model on test data 
    #test_fig, test_nc_plot = plt.subplots()
    #test_nc_plot.contourf(_xx, _yy, ncz.reshape(_xx.shape))
    #sns.scatterplot(x=test[band_1], y=test[band_2], hue=test.soil_type, palette="bright", marker='+')
    #plt.title("Nearest Centroid Visualization (Testing Data)")
    #test_plot.pyplot(test_fig)
    
    # Model Test Score
    train_plot.write(nc.score(test_bands, test_yy, sample_weight=None))


elif model_select == "Decision Trees":
    model_params, train_plot = model_prams_plots.columns([1,1])
    model_params.subheader(model_select)
    #decision_tree_descri, decision_tree_cont = model_params.columns([2, 1])
    decision_trees_form = model_params.form("Decision Tree Form")
    decision_trees_form.text("Decision Tree Form")
    critiria = decision_trees_form.selectbox("Criterion:", options=["gini", "entropy"], index=0)
    # log_loss criterion does not work Check it further!!!!!!
    max_feat = decision_trees_form.selectbox("Maximum Features:", options=["auto", "sqrt", "log2", None], index=0)
    max_depth = decision_trees_form.selectbox("Maximum Depth:", options=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50])
    decision_trees_form.form_submit_button("Apply Params")

    # Decision tree clasifier model
    decision_tree_clf = DecisionTreeClassifier(criterion=critiria, max_depth=max_depth, max_features=max_feat)
    decision_tree_clf.fit(train_bands, train_yy.ravel())
    
    # Model Params
    #st.write(decision_tree_clf.get_params())

    # Vizualization of trained model on trained data 
    _xx, _yy = viz_mesh_grid()
    cdf, cdf_arr = viz_cdf()
    decision_tree_z = decision_tree_clf.predict(cdf_arr)
    cdf['predict'] = decision_tree_z
    
    #train_plot, test_plot = st.columns([1,1])
    train_fig, train_decision_tree_plot = plt.subplots()
    train_decision_tree_plot.contourf(_xx, _yy, decision_tree_z.reshape(_xx.shape))
    sns.scatterplot(x=data[band_1], y=data[band_2], hue=data.soil_type, palette="bright", marker='+')
    plt.title("Decision Tree Visualization (Training Data)")
    train_plot.pyplot(train_fig)

    # Model train Score
    train_plot.write(decision_tree_clf.score(train_bands, train_yy, sample_weight=None))

    # Vizualization of trained model on test data 
    #test_fig, test_decision_tree_plot = plt.subplots()
    #test_decision_tree_plot.contourf(_xx, _yy, decision_tree_z.reshape(_xx.shape))
    #sns.scatterplot(x=test[band_1], y=test[band_2], hue=test.soil_type, palette="bright", marker='+')
    #plt.title("Decision Tree Visualization (Testing Data)")
    #test_plot.pyplot(test_fig)

    # Model train Score
    train_plot.write(decision_tree_clf.score(test_bands, test_yy, sample_weight=None))


    

elif model_select == "Random Forest":
    model_params, train_plot = model_prams_plots.columns([1,1])
    model_params.subheader(model_select)
    #random_forest_descri, random_forest_cont = model_params.columns([2, 1])
    random_forest_form = model_params.form("Random Forest Form")
    random_forest_form.text("Random Forest Form")
    n_estimate = random_forest_form.slider("n Estimators:", min_value=1, max_value=300, value=100, step=10)
    criteria = random_forest_form.selectbox("Criterion:", options=["gini", "entropy"], index=0)
    # log_loss does not work !!!
    max_depth = random_forest_form.selectbox("Maximum Depth:", options = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], index = 0)
    random_forest_form.form_submit_button("Apply params")

    # Random Forest Classifier Model
    rand_forest_clf = RandomForestClassifier(n_estimators=n_estimate, max_depth=max_depth, criterion=criteria)
    rand_forest_clf.fit(train_bands, train_yy.ravel())

    # Model Params
    #st.write(rand_forest_clf.get_params())

    # Vizualization of trained model on trained data 
    _xx, _yy = viz_mesh_grid()
    cdf, cdf_arr = viz_cdf()
    rand_forest_z = rand_forest_clf.predict(cdf_arr)
    cdf['predict'] = rand_forest_z
    
    #train_plot, test_plot = st.columns([1,1])
    train_fig, train_random_forest_plot = plt.subplots()
    train_random_forest_plot.contourf(_xx, _yy, rand_forest_z.reshape(_xx.shape))
    sns.scatterplot(x=data[band_1], y=data[band_2], hue=data.soil_type, palette="bright", marker='+')
    plt.title("Random Forest Visualization (Training Data)")
    train_plot.pyplot(train_fig)
    # Accuracy Assessment
    # Model Train Score
    train_plot.write(rand_forest_clf.score(train_bands, train_yy, sample_weight=None))

    # Vizualization of trained model on test data 

    #test_fig, test_random_forest_plot = plt.subplots()
    #test_random_forest_plot.contourf(_xx, _yy, rand_forest_z.reshape(_xx.shape))
    #sns.scatterplot(x=test[band_1], y=test[band_2], hue=test.soil_type, palette="bright", marker='+')
    #plt.title("Random Forest Visualization (Testing Data)")
    #test_plot.pyplot(test_fig)
    # Accuracy Assessment
    # Model Test Score
    train_plot.write(rand_forest_clf.score(test_bands, test_yy, sample_weight=None))



elif model_select == "Neural Network":
    model_params, train_plot = model_prams_plots.columns([1,1])
    model_params.subheader(model_select+": Multi-layer Perceptron classifier")
    #neural_net_descri, neural_net_cont = model_params.columns([3, 2])
    neural_net_form = model_params.form("Neural Network Form")
    neural_net_form.text("Neural Network Form")
    activation_func = neural_net_form.selectbox("Activation Functions:", options=['relu', 'identity', 'tanh', 'logistic'], index=0)
    solver_algo = neural_net_form.selectbox("Solvers:", options=['adam', 'sgd', 'lbfgs'], index=0)
    learning_rate = neural_net_form.selectbox("Learning Rate Algo:", options=['constant', 'invscaling', 'adaptive'], index=2)
    alpha = neural_net_form.select_slider("Learning Rate:", options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0], value = 0.001)
    #batch_size = neural_net_form.select_slider("Batch size:", options=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], value=64)
    hidden_layer_neurons = neural_net_form.slider("No. of neurons in Hidden Layers:", min_value=1, max_value=25, value = 10)
    depth_hidden_layer = neural_net_form.slider("Depth of Hidden Layer:", min_value = 1, max_value=25, value = 4)
    hidden_layers = (hidden_layer_neurons, depth_hidden_layer)
    max_iterations = neural_net_form.slider("Max Iterations:", min_value=1000, max_value=15000, value=5000, step=500)
    neural_net_form.form_submit_button("Apply params")

    # Neural Network - MLPClassifier
    neural_net_clf = MLPClassifier(solver=solver_algo,activation=activation_func, alpha=alpha, max_iter=max_iterations,
                           hidden_layer_sizes=hidden_layers,learning_rate=learning_rate, random_state=1)

    neural_net_clf.fit(train_bands, train_yy.ravel())
    
    # Model Params
    #st.write(neural_net_clf.get_params())

    # Vizualization of trained model on trained data 
    _xx, _yy = viz_mesh_grid()
    cdf, cdf_arr = viz_cdf()
    NN_MLP_z = neural_net_clf.predict(cdf_arr)
    cdf['predict'] = NN_MLP_z
    
    #train_plot, test_plot = st.columns([1,1])
    train_fig, train_NN_MLP_plot = plt.subplots()
    train_NN_MLP_plot.contourf(_xx, _yy, NN_MLP_z.reshape(_xx.shape))
    sns.scatterplot(x=data[band_1], y=data[band_2], hue=data.soil_type, palette="bright", marker='+')
    plt.title("Neural Network-" + activation_func + " & " + solver_algo + " (Training Data)")
    train_plot.pyplot(train_fig)
    #---- Accuracy Assessment
    # Model Train Score
    train_plot.write(neural_net_clf.score(train_bands, train_yy, sample_weight=None))

    # Vizualization of trained model on test data 
    #test_fig, test_NN_MLP_plot = plt.subplots()
    #test_NN_MLP_plot.contourf(_xx, _yy, NN_MLP_z.reshape(_xx.shape))
    #sns.scatterplot(x=test[band_1], y=test[band_2], hue=test.soil_type, palette="bright", marker='+')
    #plt.title("Neural Network-" + activation_func + " & " + solver_algo + " (Testing Data)")
    #test_plot.pyplot(test_fig)
    #---- Accuracy Assessment
    # Model Train Score
    train_plot.write(neural_net_clf.score(test_bands, test_yy, sample_weight=None))