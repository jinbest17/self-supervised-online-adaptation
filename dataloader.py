from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn import preprocessing
import tensorflow as tf

def normalize(Xmin, Xmax, X):
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            X[i][j] = (X[i][j] - Xmin[j]) / (Xmax[j] - Xmin[j])
    return X

def load_data(dataset_name, use_imb=True):
    print("loading data")
    if dataset_name == 'gas':
        data= []
        X_all = []
        y_all = []
        
        oversample = SMOTE()
        data_path = '../Dataset'
        X_y_train = load_svmlight_file(data_path + "/batch" +str(1)+".dat")
        X_train = X_y_train[0].toarray()
        y_train = X_y_train[1]
        Xmin = X_train.min(axis=0)
        Xmax = X_train.max(axis=0)
        if use_imb:
            X_train, y_train = oversample.fit_resample(X_train, y_train)
        X_y_test = load_svmlight_file(data_path + "/batch" +str(2)+".dat")
        X_test = X_y_test[0].toarray()
        y_test = X_y_test[1]
        for i in range(3,11):
            data.append(load_svmlight_file(data_path + "/batch" +str(i)+".dat"))
            X_test = np.concatenate((X_test,data[i-3][0].toarray()), axis=0)
            y_test = np.concatenate((y_test, data[i-3][1]))
        X_test = normalize(Xmin,Xmax,X_test)
        X_train = normalize(Xmin,Xmax,X_train)
        print(X_train.shape)
        print(X_test.shape)
        le = preprocessing.LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        
        return X_train, y_train, X_test, y_test
    elif dataset_name == 'mnist':
        mnist = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0

        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        le = preprocessing.LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        return X_train, y_train, X_test, y_test
                