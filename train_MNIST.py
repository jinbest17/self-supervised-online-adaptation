import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
#from tqdm.notebook import tqdm
#from sklearn.utils import shuffle
#import tensorflow_datasets as tfds
#import matplotlib.pyplot as plt
import numpy as np
import losses
#import dataloader
#from sklearn.datasets import load_svmlight_file
#import pickle
#import seaborn as sns
from collections import defaultdict
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

tf.random.set_seed(666)
np.random.seed(666)

# global configs

BS = 32
EPOCH_FEATURE = 2
EPOCH_CLASSIFIER = 5
EPOCH_FEATURE_ADAPT = 100
EPOCH_CLASSIFIER_ADAPT = 100
LOG_EVERY = 10
DATA = 'mnist-regression'
NORMALIZE_EMBEDDING = True
# NORMALIZE_EMBEDDING = False
#N_DATA_TRAIN = 60000
N_DATA_TRAIN = 800
BATCH_SIZE = 32
PROJECTION_DIM = 128
WRITE_SUMMARY = False
ACTIVATION = 'leaky_relu'
IMG_SHAPE = 28
input_shape = (28,28,1)
AUTO = tf.data.experimental.AUTOTUNE
CONFIDENCE_THRESHOLD = 0.9

class UnitNormLayer(tf.keras.layers.Layer):
    '''Normalize vectors (euclidean norm) in batch to unit hypersphere.
    '''
    def __init__(self):
        super(UnitNormLayer, self).__init__()

    def call(self, input_tensor):
        #norm = tf.norm(input_tensor, axis=1)
        #return input_tensor / tf.reshape(norm, [-1, 1])
        return tf.math.l2_normalize(input_tensor, axis=1)

def print_batch_size(samples):
    for key in samples:
        print(key,' : ',len(samples[key]))

        
# https://stackoverflow.com/questions/50127257/is-there-any-way-to-stop-training-a-model-in-keras-after-a-certain-accuracy-has/50127361       
class EarlyStoppingByAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, monitor='sparse_categorical_accuracy', value=0.995, verbose=0):
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


def get_threshold_mnist(X_train_proj, y_train):
    class_count_train = Counter(y_train)

    def def_value_b():
        return np.zeros(X_train_proj[0].shape)
    train_proj_by_class = defaultdict(def_value_b)
    for i in range(1, len(X_train)):
        train_proj_by_class[y_train[i]] += X_train_proj[i]
    for i in range(0,10):
        train_proj_by_class[i] = train_proj_by_class[i] / class_count_train[i]
    distance_train = defaultdict(list)
    avg_distance_train = {}
    for i in range(len(X_train_proj)):
        distance_train[y_train[i]].append(np.linalg.norm(X_train_proj[i] - train_proj_by_class[y_train[i]]))

    for i in range(0,10):
        avg_distance_train[i] = np.mean(distance_train[i])
    return avg_disatnce_train, train_proj_by_class

def retrain_contrastive(encoder_r, projector_z, train_ds,optimizer3):
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            r = encoder_r(images, training=True)
            z = projector_z(r, training=True)
            loss = losses.max_margin_contrastive_loss(z, labels)

        gradients = tape.gradient(loss, 
            encoder_r.trainable_variables + projector_z.trainable_variables)
        optimizer3.apply_gradients(zip(gradients, 
            encoder_r.trainable_variables + projector_z.trainable_variables))

        return loss
    
    
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
def train_mnist_offline(X_train, y_train): 
    
    # simulate low data regime for training
    n_train = X_train.shape[0]
    shuffle_idx = np.arange(n_train)
    np.random.shuffle(shuffle_idx)

    X_train_small = X_train[shuffle_idx][:N_DATA_TRAIN]
    y_train_small = y_train[shuffle_idx][:N_DATA_TRAIN]
    print(X_train_small.shape, y_train_small.shape)
    
    # train the model
    train_ds=tf.data.Dataset.from_tensor_slices((X_train,y_train))
  
    train_ds = (
        train_ds
        .shuffle(100)
        .batch(BS)
        .prefetch(AUTO)
    )


    optimizer3=tf.keras.optimizers.Adam(learning_rate=0.0005 )
    encoder_r = encoder_net()
    projector_z = projector_net()

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            r = encoder_r(images, training=True)
            z = projector_z(r, training=True)
            loss = losses.max_margin_contrastive_loss(z, labels)

        gradients = tape.gradient(loss, 
            encoder_r.trainable_variables + projector_z.trainable_variables)
        optimizer3.apply_gradients(zip(gradients, 
            encoder_r.trainable_variables + projector_z.trainable_variables))

        return loss
    train_loss_results = []
    # train contrastive feature
    for epoch in range(EPOCH_FEATURE):	
        epoch_loss_avg = tf.keras.metrics.Mean()
        
        for (images, labels) in train_ds:
            loss = train_step(images, labels)
            epoch_loss_avg.update_state(loss) 
        #if epoch_loss_avg.result() < 0.004:
            #print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
            #print("Encoder train exiting")
            #break        
        train_loss_results.append(epoch_loss_avg.result())
        
        if epoch % LOG_EVERY == 0:
            print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

    # Train classifier
    def supervised_model():
    
        inputs = Input(input_shape)
        encoder_r.trainable = False
        projector_z.trainable = False
        r = projector_z(encoder_r(inputs, training=False), training=False)
        outputs = Dense(10, activation='softmax')(r)

        supervised_model = Model(inputs, outputs)
        return supervised_model

    optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    supervised_classifier = supervised_model()

    supervised_classifier.compile(optimizer=optimizer2,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    supervised_classifier.fit(train_ds,
        epochs=EPOCH_CLASSIFIER,
        verbose=1)
    return supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small
    
    

def train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test, train_proj_by_class,avg_distance_train,verbose=True):
    # @tf.function
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:
    #         r = encoder_r(images, training=True)
    #         z = projector_z(r, training=True)
    #         loss = losses.max_margin_contrastive_loss(z, labels)

    #     gradients = tape.gradient(loss, 
    #         encoder_r.trainable_variables + projector_z.trainable_variables)
    #     optimizer3.apply_gradients(zip(gradients, 
    #         encoder_r.trainable_variables + projector_z.trainable_variables))

    #     return loss
    results = []
    rus = RandomUnderSampler()
    X_target = []
    y_target = []
    BATCH_SIZE = 32
    NUM_TEST = 3200
    BS_ADAPT = 300
    NUM_BATCH = NUM_TEST // BS_ADAPT
    for i in range(0,NUM_BATCH):
        X_batch = X_test[i*BS_ADAPT:(i+1)*BS_ADAPT].reshape(BS_ADAPT,28,28,1)
        batch_proj = projector_z.predict(encoder_r.predict(X_batch))
        sample_score = supervised_classifier.predict(X_batch)
        labels = sample_score.argmax(axis=1) 
        print("training batch "+str(i*BS_ADAPT))
        results = np.concatenate((results, labels))    
        X_target = []
        y_target = []
        for j in range(0, BS_ADAPT):
            d = np.linalg.norm(batch_proj[j] - train_proj_by_class[labels[j]])
            if d < avg_distance_train[labels[j]] + 0.35:
                X_target.append(X_batch[j])
                y_target.append(labels[j])
        #print(Counter(y_target))
        X_target = np.array(X_target).reshape(len(y_target), -1)
        #print(X_target.shape)

        y_target = np.array(y_target)
        
        X_target, y_target = rus.fit_resample(X_target,y_target)
        print(Counter(y_target))

        #print(X_target.shape, y_target.shape)
        X_target = X_target.reshape(len(y_target), 28,28,1)
        train_ds=tf.data.Dataset.from_tensor_slices((X_target,y_target))
        train_ds = (
          train_ds
          .shuffle(100)
          .batch(BATCH_SIZE)
          .prefetch(AUTO)
        )
        
        retrain_contrastive(encoder_r, projector_z, train_ds,optimizer3)
        encoder_r.trainable = False
        projector_z.trainable = False
        supervised_classifier.fit(train_ds,
            epochs=3)
        
    return results

def load_model():
    optimizer3=tf.keras.optimizers.Adam(learning_rate=0.0005 )
    encoder_r = encoder_net()
    projector_z = projector_net()
    def supervised_model():
    
        inputs = Input(input_shape)
        encoder_r.trainable = False
        projector_z.trainable = False
        r = projector_z(encoder_r(inputs, training=False), training=False)
        outputs = Dense(10, activation='softmax')(r)

        supervised_model = Model(inputs, outputs)
        return supervised_model

    optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    supervised_classifier = supervised_model()

    supervised_classifier.compile(optimizer=optimizer2,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    encoder_r.load_weights("./SCL_encoder_MNIST_11.h5")
    projector_z.load_weights("./SCL_projector_MNIST_11.h5")
    supervised_classifier.load_weights('./Classifier_MNIST_11.h5')
    
    return supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3
    
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