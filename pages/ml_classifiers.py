from decimal import MIN_EMIN
from pyexpat import model
from click import option
from requests import options
import streamlit as st

from pandas import read_csv
from pandas import DataFrame

from sklearn.neighbors import KNeighborsClassifier

from numpy import arange
from numpy import meshgrid
from numpy import array

from seaborn import scatterplot
from matplotlib import pyplot as plt


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

print(data)
# Sidebar
st.sidebar.header("Parms")

model_select = st.sidebar.selectbox(label="Select ML Classifier:", options=["KNN Classifier", "Nearest Centroid", "Decision Trees", "Random Forest", "Neural Network"])

band_select = st.sidebar.form("select_band")
band_1 = band_select.selectbox("Select Band 1:", options=['b1', 'b2', 'b3', 'b4'], index = 2)
band_2 = band_select.selectbox("Select Band 2:", options=['b1', 'b2', 'b3', 'b4'], index = 3)
band_select.form_submit_button("Apply Bands")

model_params = st.container()

# selecting bands for model training 

train_bands = array(data[[band_1, band_2]])
test_bands = array(test[[band_1, band_2]])

train_yy = array(data[['soil_type']])
test_yy = array(test[['soil_type']])



if model_select == "KNN Classifier":
    model_params.subheader(model_select)
    knn_model_descri, knn_model_cont = model_params.columns([2, 1])
    knn_params_form = knn_model_cont.form("KNN Params Form")
    knn_params_form.text("KNN Params Form")
    n_ngbrs = knn_params_form.slider("N Nearest Neighbours:", min_value=1, max_value=100, value=50, step=5)
    weights = knn_params_form.selectbox("Weight Function:" ,options=["uniform", "distance"], index=0)
    algo = knn_params_form.selectbox("Nearest Neighbour Algorithm:", options=['auto', 'ball_tree', 'kd_tree', 'brute'], index=0)
    knn_params_form.form_submit_button("Apply Params")

    # initiate Model classifier
    knn = KNeighborsClassifier(n_neighbors=n_ngbrs, weights=weights, algorithm=algo)
    knn.fit(train_bands, train_yy.ravel())

    # Model Params
    st.write(knn.get_params())
    # Model Score
    st.write(knn.score(test_bands, test_yy, sample_weight=None))

    # Vizualization
    _xx, _yy = viz_mesh_grid()
    cdf, cdf_arr = viz_cdf()
    knnz = knn.predict(cdf_arr)
    cdf['predict'] = knnz
    #st.dataframe(cdf)
    fig, ax = plt.subplots()
    ax.contourf(_xx, _yy, knnz.reshape(_xx.shape))
    scatterplot(x=data[band_1], y=data[band_2], hue=data.soil_type, palette="deep", marker='+')
    plt.title("KKN Visualization")
    st.pyplot(fig)
    


elif model_select == "Nearest Centroid":
    
    pass

elif model_select == "Decision Trees":
    model_params.subheader(model_select)
    decision_tree_descri, decision_tree_cont = model_params.columns([2, 1])
    decision_trees_form = decision_tree_cont.form("Decision Tree Form")
    decision_trees_form.text("Decision Tree Form")
    critiria = decision_trees_form.selectbox("Criterion:", options=["gini", "entropy", "log_loss"], index=0)
    max_feat = decision_trees_form.selectbox("Maximum Features:", options=["auto", "sqrt", "log2", None], index=0)
    max_depth = decision_trees_form.selectbox("Maximum Depth:", options=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    decision_trees_form.form_submit_button("Apply Params")

elif model_select == "Random Forest":
    model_params.subheader(model_select)
    random_forest_descri, random_forest_cont = model_params.columns([2, 1])
    random_forest_form = random_forest_cont.form("Random Forest Form")
    random_forest_form.text("Random Forest Form")
    n_estimate = random_forest_form.slider("n Estimators:", min_value=10, max_value=500, value=100, step=10)
    criteria = random_forest_form.selectbox("Criterion:", options=["gini", "entropy", "log_loss"], index=0)
    max_depth = random_forest_form.selectbox("Maximum Depth:", options = [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], index = 0)

    random_forest_form.form_submit_button("Apply params")

elif model_select == "Neural Network":
    model_params.subheader(model_select+": Multi-layer Perceptron classifier")
    neural_net_descri, neural_net_cont = model_params.columns([3, 2])
    neural_net_form = neural_net_cont.form("Neural Network Form")
    neural_net_form.text("Neural Network Form")
    activation_func = neural_net_form.selectbox("Activation Functions:", options=['relu', 'identity', 'tanh', 'logistic'], index=0)
    solver_algo = neural_net_form.selectbox("Solvers:", options=['adam', 'sgd', 'lbfgs'], index=0)
    learn_rate = neural_net_form.selectbox("Learning Rate:", options=['constant', 'invscaling', 'adaptive'], index=0)
    hidden_layer_neurons = neural_net_form.slider("Hidden Layer neurons:", min_value=1, max_value=25, value = 10)
    depth_hidden_layer = neural_net_form.slider("Depth of Hidden Layer:", min_value = 1, max_value=10, value = 4)
    _hidden_layers = (hidden_layer_neurons, depth_hidden_layer)
    neural_net_form.form_submit_button("Apply params")



