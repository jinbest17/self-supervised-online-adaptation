import numpy as np
from numpy.random import seed
from numpy.random import randint
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
from sklearn.utils import random
from lssvm import LSSVC
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

import pandas as pd


def train_baseline_lssvm_offline(X, y):
    clusters = 15
    kmeans = KMeans(n_clusters=clusters).fit(X)


    clusterList = defaultdict(list)

    for i in range(len(y)):
        clusterList[kmeans.labels_[i]].append((X[i], y[i]))

    X_train = []
    y_train = []

    for key in clusterList:
        sampleIdx = random.sample_without_replacement(n_population=len(clusterList[key]),  n_samples=len(clusterList[key])//2)
        for j in sampleIdx:
            X_train.append(clusterList[key][j][0])
            y_train.append(clusterList[key][j][1])
    X_train, y_train = shuffle(X_train, y_train)
    
    model = LSSVC(gamma=1, kernel='rbf', l = 5000000, dTh=0.9, sigma=1) # Class instantiation

    model.fit(X_train, np.array(y_train)) # Fitting the model
    
    return model
def train_baseline_lssvm_online(X_test,model):
    return model.fit_online(X_test[:1245])

def evaluate(y_true, y_pred):
    batch_length = [445, 1244, 1586,161,197,2300,3613,294,470,3600]
    countnum = 0
    counttotal = 0
    count = 1
    i = 0
    while count < 10:
        
        if y_true[i] == y_pred[i]:
            countnum+=1
        counttotal += 1
        i+= 1
        if counttotal == batch_length[count]:
            print("Batch", count+1, " Accuracy is: ", countnum/counttotal)
            countnum = 0
            counttotal = 0
            count+= 1