import os
os.environ['WANDB_DISABLE_CODE'] = 'True'
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.utils import shuffle
from sklearn import preprocessing
import numpy as np
import losses
import ot
from sklearn.datasets import load_svmlight_file
import pickle

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from collections import defaultdict
import sys
import time
import socket

UDP_IP = "169.254.200.28"
UDP_PORT = 30000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# global configs
EPOCHS = 20
DATA = 'mnist-regression'
NORMALIZE_EMBEDDING = True
# NORMALIZE_EMBEDDING = False
#N_DATA_TRAIN = 60000
N_DATA_TRAIN = 800
BATCH_SIZE = 64
PROJECTION_DIM = 128
WRITE_SUMMARY = False
ACTIVATION = 'leaky_relu'
IMG_SHAPE = 32
input_shape = (32,32,3)
rus = RandomUnderSampler()
AUTO = tf.data.experimental.AUTOTUNE

class EarlyStoppingByAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, monitor='sparse_categorical_accuracy', value=0.985, verbose=0):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def FNN():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32,32,3)))
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

model = FNN()
model.load_weights("./Classifier_CIFAR_OT.h5")
with open("cifar_ot.p", "rb") as handler:
    X_train_small = pickle.load(handler)

ot_model = ot.da.EMDTransport()
X_test = np.load("fog.npy")
X_source = X_train_small['X_train_small'].reshape(800,-1)
X_target = []
y_target = []
BATCH_SIZE = 64
NUM_TEST = 3200
BS_ADAPT = 100
NUM_BATCH = NUM_TEST // BS_ADAPT
#start online adaptation
sock.sendto(b's,cifar_ot', (UDP_IP, UDP_PORT))


st = time.time()
results = []

for i in range(0,NUM_BATCH):
    print("num adapted: "+str(i*BS_ADAPT))
    X_batch = X_test[i*BS_ADAPT:(i+1)*BS_ADAPT].reshape(BS_ADAPT,32,32,3)
    if i>0:
        label =  np.argmax(model.predict(ot_model.transform(X_batch.reshape((BS_ADAPT,-1))).reshape(BS_ADAPT,32,32,3)),axis=1)
    else:
        label =  np.argmax(model.predict(X_batch),axis=1)
    print(label)
    results= np.concatenate((results, label))
    X_target = X_batch
    y_target = label
    X_target = np.array(X_target)
    X_target,y_target = shuffle(X_target,y_target)
    X_target_reshaped = X_target.reshape((BS_ADAPT,-1))
    X_target_reshaped,y_target = rus.fit_resample(X_target_reshaped,y_target)
    print(X_target_reshaped.shape)
    ot_model.fit(Xs=X_target_reshaped, Xt=X_source)
    X_target_reshaped = np.asarray(X_target_reshaped).astype('float32')
    y_target = np.asarray(y_target).astype('int32')
    model.fit(X_target_reshaped.reshape(-1,32,32,3),y_target, batch_size=80,epochs=50, callbacks=[EarlyStoppingByAccuracy()])
    X_source = X_target_reshaped

ed = time.time()
sock.sendto(b't,', (UDP_IP, UDP_PORT))
with open('time_log.txt', 'a+') as f:
    f.write('gas SOA online exec time: {} secs\n'.format(ed - st))

