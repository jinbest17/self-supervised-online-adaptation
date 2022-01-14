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
IMG_SHAPE = 32
input_shape = (32,32,1)
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
    
    

def train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test, verbose=False):
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
    # Calculate score for each class in train
    score_batch1 = supervised_classifier.predict(X_train_small)
    score_max_1 = score_batch1.max(axis=1)
    scores_per_class = defaultdict(list)
    for i in range(len(y_train_small)):
        scores_per_class[y_train_small[i]].append(score_max_1[i])
    mean_score = {key:np.mean(scores_per_class[key]) for key in scores_per_class}
    print(mean_score)

    sample_per_class = defaultdict(list)
    for i in range(len(y_train_small)):
        sample_per_class[y_train_small[i]].append(X_train_small[i])
    for key in sample_per_class:
        print(len(sample_per_class[key]))
    
    results = []
    #max_scores = []

    count = 0
    BATH_SIZE_ADAPT = 100
    new_samples_dict = defaultdict(list)
    accumulate_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
    lowerbound = {key: mean_score[key] * CONFIDENCE_THRESHOLD for key in mean_score}
    
    for i in range(0,len(X_test)):
        
        sample_score = supervised_classifier.predict(X_test[i].reshape(1,32,32,1))
        label = sample_score.argmax() 
        results.append(label)    
        max_score = sample_score.max()
        #max_scores.append(max_score)
        if max_score < mean_score[label]  and  max_score > lowerbound[label]:
            # add to new training sample
            if accumulate_count[label] >= len(sample_per_class[label]):
                accumulate_count[label] = 0
            sample_per_class[label][accumulate_count[label]] = X_test[i]
            accumulate_count[label] += 1
            count+=1
            
            
        # if we accumulate enough samples
        if count >= BATH_SIZE_ADAPT:
            print("New adaptation initiated at sample", i)
            print_batch_size(sample_per_class)
            # form target batch
            X_target =[]
            y_target = []

            for key in sample_per_class:
                X_target = X_target + sample_per_class[key]
                y_target = np.concatenate((y_target,[key]*len(sample_per_class[key])))        
            
            X_target = np.array(X_target)

            print(X_target.shape, y_target.shape)
            train_ds=tf.data.Dataset.from_tensor_slices((X_target,y_target))
            train_ds = (
                train_ds
                .shuffle(100)
                .batch(BS)
                .prefetch(AUTO)
            )
            
            train_loss_results = []
            encoder_r.trainable = True
            projector_z.trainable = True
            for epoch in range(EPOCH_FEATURE_ADAPT):	
                epoch_loss_avg = tf.keras.metrics.Mean()

                for (images, labels) in train_ds:
                    loss = train_step(images, labels)
                    epoch_loss_avg.update_state(loss) 

                train_loss_results.append(epoch_loss_avg.result())
                if epoch_loss_avg.result() < 0.004:
                    print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
                    print("Encoder train exiting")
                    break
                if epoch % LOG_EVERY == 0:
                    print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
            encoder_r.trainable = False
            projector_z.trainable = False
            supervised_classifier.fit(train_ds,
                epochs=EPOCH_CLASSIFIER_ADAPT, verbose=1,callbacks=[EarlyStoppingByAccuracy()])
            
            count = 0
            scores_per_class = defaultdict(list)
            score_batch1 = supervised_classifier.predict(X_target)
            score_max_1 = score_batch1.max(axis=1)
            for i in range(len(y_target)):
                scores_per_class[y_target[i]].append(score_max_1[i])
            mean_score = {key:np.mean(scores_per_class[key]) for key in scores_per_class}
            print(mean_score)
            lowerbound = {key: mean_score[key] * CONFIDENCE_THRESHOLD for key in mean_score}
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