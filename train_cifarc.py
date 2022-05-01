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

from sklearn.datasets import load_svmlight_file
import pickle


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
AUTO = tf.data.experimental.AUTOTUNE
# Reference: https://github.com/wangz10/contrastive_loss/blob/master/model.py
class UnitNormLayer(tf.keras.layers.Layer):
    '''Normalize vectors (euclidean norm) in batch to unit hypersphere.
    '''
    def __init__(self):
        super(UnitNormLayer, self).__init__()

    def call(self, input_tensor):
        norm = tf.norm(input_tensor, axis=1)
        return input_tensor / tf.reshape(norm, [-1, 1])
# Encoder Network
def encoder_net():
	#inputs = Input((IMG_SHAPE, IMG_SHAPE, 1))

	#normalization_layer = UnitNormLayer()
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(UnitNormLayer())
	#encoder = tf.keras.applications.ResNet50(weights=None, include_top=False)
	model.trainable = True

	#embeddings = encoder(inputs, training=True)
	#embeddings = GlobalAveragePooling2D()(embeddings)
	#norm_embeddings = normalization_layer(embeddings)

	#encoder_network = Model(inputs, norm_embeddings)

	return model

# Projector Network
def projector_net():
	projector = tf.keras.models.Sequential([
		Input(shape=(512,)),
		Dense(256, activation="relu"),
		UnitNormLayer()
	])

	return projector
def supervised_model():
    
    inputs = Input(input_shape)
    encoder_r.trainable = False
    projector_z.trainable = False
    r = projector_z(encoder_r(inputs, training=False), training=False)
    outputs = Dense(10, activation='softmax')(r)

    supervised_model = Model(inputs, outputs)
    return supervised_model

def retrain_contrastive(encoder_r, projector_z, train_ds):
  EPOCHS =20
  LOG_EVERY = 10
  train_loss_results = []
  encoder_r.trainable = True
  projector_z.trainable = True
  for epoch in range(EPOCHS):	
      epoch_loss_avg = tf.keras.metrics.Mean()

      for (images, labels) in train_ds:
          loss = train_step(images, labels)
          epoch_loss_avg.update_state(loss) 

      train_loss_results.append(epoch_loss_avg.result())
      if epoch_loss_avg.result() < 0.005:
          print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
          print("Encoder train exiting")
          break
      if epoch % LOG_EVERY == 0:
          print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))


@tf.function
def train_step(images, labels):
	with tf.GradientTape() as tape:
		r = encoder_r(images, training=True)
		z = projector_z(r, training=True)
		#print(z.shape)
		loss = losses.max_margin_contrastive_loss(z, labels)
	gradients = tape.gradient(loss, 
		encoder_r.trainable_variables + projector_z.trainable_variables)
	optimizer3.apply_gradients(zip(gradients, 
		encoder_r.trainable_variables + projector_z.trainable_variables))

	return loss

sock.sendto(b's,gas_SOA', (UDP_IP, UDP_PORT))

optimizer3=tf.keras.optimizers.Adam(learning_rate=0.001 )
encoder_r = encoder_net()
projector_z = projector_net()
optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-3)
supervised_classifier = supervised_model()

supervised_classifier.compile(optimizer=optimizer2,
	loss=tf.keras.losses.SparseCategoricalCrossentropy(),
	metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

encoder_r.load_weights("./SCL_encoder_CIFAR_11.h5")
projector_z.load_weights("./SCL_projector_CIFAR_11.h5")
supervised_classifier.load_weights('./Classifier_CIFAR_11.h5')

X_test = np.load("fog.npy")
with open("class_center.p","rb") as handler:
    class_center = pickle.load(handler)

avg_distance_train = class_center['avg_distance_train']
train_proj_by_class = class_center['train_proj_by_class']

st = time.time()


X_target = []
y_target = []
BATCH_SIZE = 64
NUM_TEST = 3200
BS_ADAPT = 400
NUM_BATCH = NUM_TEST // BS_ADAPT
for i in range(0,NUM_BATCH):
    
    X_batch = X_test[i*BS_ADAPT:(i+1)*BS_ADAPT].reshape(BS_ADAPT,32,32,3)
    batch_proj = projector_z.predict(encoder_r.predict(X_batch))
    sample_score = supervised_classifier.predict(X_batch)
    labels = sample_score.argmax(axis=1) 
    print(k)
    results_all[k] = np.concatenate((results_all[k], labels))    
    X_target = []
    y_target = []
    for j in range(0, BS_ADAPT):
        d = np.linalg.norm(batch_proj[j] - train_proj_by_class[labels[j]])
        if d < avg_distance_train[labels[j]] + 0.4:
            X_target.append(X_batch[j])
            y_target.append(labels[j])
    #print(Counter(y_target))
    X_target = np.array(X_target).reshape(len(y_target), -1)
    y_target = np.array(y_target)
      
    X_target, y_target = rus.fit_resample(X_target,y_target)
    #print(Counter(y_target))
    X_target = X_target.reshape(len(y_target), 32,32,3)

    X_target = np.asarray(X_target).astype('float32')
    y_target = np.asarray(y_target).astype('int32')
    train_ds=tf.data.Dataset.from_tensor_slices((X_target,y_target))
    train_ds = (
        train_ds
        .shuffle(100)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    retrain_contrastive(encoder_r, projector_z, train_ds)
    encoder_r.trainable = False
    projector_z.trainable = False
    supervised_classifier.fit(train_ds,
        epochs=3)

ed = time.time()
sock.sendto(b't,', (UDP_IP, UDP_PORT))
with open('time_log.txt', 'a+') as f:
    f.write('gas SOA online exec time: {} secs\n'.format(ed - st))

