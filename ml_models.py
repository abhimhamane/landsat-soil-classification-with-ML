from pandas import DataFrame
import numpy as np


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

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

def max_depth_decision_tree(critiria, max_depth, max_feat):
    dc_model = DecisionTreeClassifier(criterion=critiria, max_depth=max_depth, max_features=max_feat)