from collections import defaultdict
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.utils import shuffle
import numpy as np
#import losses
import pickle
tf.random.set_seed(666)
np.random.seed(666)
import ot
import dataloader


def FNN():
    inputs = Input((128,))
    FNN = tf.keras.models.Sequential([
        Dense(100, activation="relu"),
        Dense(80,activation="relu"),
        Dense(50,activation="relu"),
        Dense(6,activation="softmax")
    ])
    embeddings = FNN(inputs, training=True)
    FNN_network = Model(inputs, embeddings)
    #decay = tf.train
    FNN_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return FNN_network

def train_gas_ot_offline(X_train, y_train):
    model = FNN()
    model.fit(X_train,y_train, batch_size=80,epochs=20)

def train_gas_ot_online(X_train, y_train, X_test, model):
    ot_model=ot.da.EMDTransport()
    results = []
    X_source = X_train
    sample_per_class = defaultdict(list)
    accumulate_count = {1:0,2:0,3:0,4:0,5:0,0:0}
    new_samples_dict = defaultdict(list)
    BATCH_SIZE_ADAPT = 100


    for i in range(len(y_train)):
        sample_per_class[y_train[i]].append(X_train[i])
    count = 0
    for i in range(0,len(X_test)):
        label =  np.argmax(model.predict(X_test[i].reshape(1,128)))
        if accumulate_count[label] >= len(sample_per_class[label]):
            accumulate_count[label] = 0
        sample_per_class[label][accumulate_count[label]] = X_test[i]
        accumulate_count[label] += 1
        count+=1
        if count > BATCH_SIZE_ADAPT:
            X_target = []
            y_target = []
            for key in sample_per_class:
                X_target = X_target + sample_per_class[key]
                y_target = np.concatenate((y_target,[key]*len(sample_per_class[key])))
            X_target = np.array(X_target)
            X_target,y_target = shuffle(X_target,y_target)
            ot_model.fit(Xs=X_target, Xt=X_source)
            X_target_transform = ot_model.transform(Xs = X_target)

            model.fit(X_target,y_target, batch_size=80,epochs=3)
            count = 0
            print(i)
        results.append(label)
    return results
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
    
