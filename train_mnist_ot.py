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
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    #encoder = tf.keras.applications.ResNet50(weights=None, include_top=False)
    model.trainable = True
    #decay = tf.train
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

def train_mnist_ot_offline(X_train, y_train):
    model = FNN()
    model.fit(X_train,y_train, batch_size=32,epochs=5)
    #model.save_weights("./Classifier_MNIST_OT.h5")
    return model
def load_model():
    model = FNN()
    model.load_weights("./Classifier_MNIST_OT.h5")
    return model
def train_mnist_ot_online(X_train, y_train, X_test, model):
    ot_model=ot.da.EMDTransport()
    results = []
    X_source = X_train
    sample_per_class = defaultdict(list)
    accumulate_count = {1:0,2:0,3:0,4:0,5:0,0:0,6:0,7:0,8:0,9:0}
    new_samples_dict = defaultdict(list)
    BATCH_SIZE_ADAPT = 100


    for i in range(len(y_train)):
        sample_per_class[y_train[i]].append(X_train[i])
    count = 0
    for i in range(0,len(X_test)):
        label =  np.argmax(model.predict(X_test[i].reshape(1,28,28,1)))
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

            model.fit(X_target,y_target, batch_size=32,epochs=1)
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
    
