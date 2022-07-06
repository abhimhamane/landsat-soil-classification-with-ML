from tracemalloc import take_snapshot
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



def n_estim_knn(weights, algo, train_bands, train_yy, test_bands, test_yy):
    n_estimators = np.arange(1, 101)
    n_train_score = []
    n_test_score = []
    for est in n_estimators:
        knn = KNeighborsClassifier(n_neighbors=est, weights=weights, algorithm=algo)
        knn.fit(train_bands, train_yy.ravel())
        n_train_score.append(knn.score(train_bands, train_yy, sample_weight=None))
        n_test_score.append(knn.score(test_bands, test_yy, sample_weight=None))
    
    return n_estimators, n_train_score, n_test_score

def max_depth_decision_tree(critiria, max_feat, train_bands, train_yy ,test_bands, test_yy):
    depths = np.arange(1, 51)
    md_train_score = []
    md_test_score= []
    for md in depths:
        decision_tree_clf = DecisionTreeClassifier(criterion=critiria, max_depth=md, max_features=max_feat)
        decision_tree_clf.fit(train_bands, train_yy.ravel())
        md_train_score.append(decision_tree_clf.score(train_bands, train_yy, sample_weight=None))
        md_test_score.append(decision_tree_clf.score(test_bands, test_yy, sample_weight=None))
    
    return depths, md_train_score, md_test_score

def activation_func_plot(solver_algo, alpha, max_iterations, hidden_layers, learning_rate, train_bands, train_yy, test_bands, test_yy):
    activation_func = ['relu', 'tanh', 'identity', 'logistic']

    act_score_train = []
    act_score_test = []
    for act_func in activation_func:
        neural_net_clf = MLPClassifier(solver=solver_algo,activation=act_func, alpha=alpha, max_iter=max_iterations,
                            hidden_layer_sizes=hidden_layers,learning_rate=learning_rate, random_state=1)
        
        neural_net_clf.fit(train_bands, train_yy.ravel())
        act_score_train.append(neural_net_clf.score(train_bands, train_yy, sample_weight=None))
        act_score_test.append(neural_net_clf.score(test_bands, test_yy, sample_weight=None))

    return activation_func, act_score_train, act_score_test

    
    

def hidden_layers_depth(num_neurons):
    layers = []
    dpth = []
    for depth in range(1, 26):
        layers.append((num_neurons, depth))
        dpth.append(depth)
    return layers, dpth


def hidden_layer_MLP_plot():

    pass
    
def relu_depth_nn(solver_algo, alpha, max_iterations, learning_rate, neurons_in_layer, train_bands, train_yy, test_bands, test_yy):
    act_func = 'relu'
    act_score_train = []
    act_score_test = []
    depths = np.arange(1, 26, 1)
    for depth in range(1, 26):
        hidden_layer = (neurons_in_layer, depth)
        neural_net_clf = MLPClassifier(solver=solver_algo,activation=act_func, alpha=alpha, max_iter=max_iterations,
                            hidden_layer_sizes=hidden_layer,learning_rate=learning_rate, random_state=1)
        
        neural_net_clf.fit(train_bands, train_yy.ravel())
        act_score_train.append(neural_net_clf.score(train_bands, train_yy, sample_weight=None))
        act_score_test.append(neural_net_clf.score(test_bands, test_yy, sample_weight=None))
    
    fig, ax = plt.subplots()
    ax.plot(depths, act_score_train, label='Train - ReLU '+solver_algo, color='r')
    ax.plot(depths, act_score_test, label='Test - ReLU ' + solver_algo, color='g')
    ax.legend()
    ax.xlabel()
    ax.title("")
    return fig
    







