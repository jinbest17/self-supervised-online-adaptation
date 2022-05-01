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
tf.random.set_seed(666)
np.random.seed(666)

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
    for key in range(0,6):
        print(key,' : ',len(samples[key]))

        
# https://stackoverflow.com/questions/50127257/is-there-any-way-to-stop-training-a-model-in-keras-after-a-certain-accuracy-has/50127361       
class EarlyStoppingByAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, monitor='sparse_categorical_accuracy', value=0.98, verbose=0):
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

def encoder_net():
	inputs = Input((128,))
	normalization_layer = UnitNormLayer()

	encoder = tf.keras.models.Sequential([
		Dense(100, activation="relu"),
		UnitNormLayer()
	])
	encoder.trainable = True

	embeddings = encoder(inputs, training=True)
	norm_embeddings = normalization_layer(embeddings)

	encoder_network = Model(inputs, norm_embeddings)

	return encoder_network

# Projector Network
def projector_net():
    inputs = Input((100,))
    layer = tf.keras.models.Sequential([
		Dense(80, activation="relu",input_shape=(128,1)),
		UnitNormLayer()
	])
    
    return layer


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
    
    
    EPOCHS =40
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

def train_gas_offline(X_train, y_train, X_test, y_test, verbose=True): 
    IMG_SHAPE = 128
    BS = 80
    EPOCH_FEATURE = 100
    EPOCH_CLASSIFIER = 100
    EPOCH_FEATURE_ADAPT = 100
    EPOCH_CLASSIFIER_ADAPT = 100
    LOG_EVERY = 10
    AUTO = tf.data.experimental.AUTOTUNE
    CONFIDENCE_THRESHOLD = 0.8
    batch_length = [445, 1244, 1586,161,197,2300,3613,294,470,3600]
    # train the model
    train_ds=tf.data.Dataset.from_tensor_slices((X_train,y_train))
    validation_ds=tf.data.Dataset.from_tensor_slices((X_test,y_test))
  
    train_ds = (
        train_ds
        .shuffle(100)
        .batch(BS)
        .prefetch(AUTO)
    )
    validation_ds = (
        validation_ds
        .shuffle(100)
        .batch(BS)
        .prefetch(AUTO)
    )

    optimizer3=tf.keras.optimizers.Adam(learning_rate=0.0003 )
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
        #if epoch_loss_avg.result() < 0.001:
        
        train_loss_results.append(epoch_loss_avg.result())
        if epoch_loss_avg.result() < 0.004:
            break
        if verbose == True and epoch % LOG_EVERY == 0:
            print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
    #print(train_loss_results[-10:])

    # Train classifier
    def supervised_model():
    
        inputs = Input((IMG_SHAPE,  ))
        encoder_r.trainable = False
        projector_z.trainable = False
        r = projector_z(encoder_r(inputs, training=False), training=False)
        #r = encoder_r(inputs, training=False)
        outputs = Dense(6, activation='softmax')(r)

        supervised_model = Model(inputs, outputs)
        return supervised_model
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    supervised_classifier = supervised_model()

    supervised_classifier.compile(optimizer=optimizer2,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    supervised_classifier.fit(train_ds,
        epochs=EPOCH_CLASSIFIER,
        verbose=1 if verbose else 0)
    count = 0
    print("No transfer Accuracy")
    for i in range(1,10):
        print(supervised_classifier.evaluate(X_test[count:count+batch_length[i]],y_test[count:count+batch_length[i]]))
        count += batch_length[i]
    
    
    return supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3

def train_gas_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train, y_train, X_test, y_test, verbose=True):
    EPOCH_FEATURE_ADAPT = 25
    EPOCH_CLASSIFIER_ADAPT = 60
    LOG_EVERY = 10
    BS = 80
    AUTO = tf.data.experimental.AUTOTUNE
    NUM_TEST = len(X_test)
    BS_ADAPT = 200
    NUM_BATCH = NUM_TEST // BS_ADAPT
    BATCH_SIZE_ADAPT = 20
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
   
        
    # Set samples for each class in train
    sample_per_class = [[],[],[],[],[],[]]
    for i in range(len(y_train)):
        sample_per_class[y_train[i]].append(X_train[i])

    results = []
    count = 0
    BATH_SIZE_ADAPT = 60
    accumulate_count = [0,0,0,0,0,0]
    
    for i in range(0,NUM_BATCH+1):
        
        X_batch = X_test[i*BS_ADAPT:(i+1)*BS_ADAPT]
        batch_proj = projector_z.predict(encoder_r.predict(X_batch))
        sample_score = supervised_classifier.predict(X_batch)
        labels = sample_score.argmax(axis=1) 
        results = np.concatenate((results, labels))  
    
        
        for j in range(0, len(X_batch)):
            d = np.linalg.norm(batch_proj[j] - train_proj_by_class[labels[j]])
            if d < avg_distance_train[labels[j]] + 0.7:
                #X_target.append(X_batch[j])
                #y_target.append(labels[j])
                if accumulate_count[labels[j]] >= len(sample_per_class[labels[j]]):
                    accumulate_count[labels[j]] = 0
                sample_per_class[labels[j]][accumulate_count[labels[j]]] = X_batch[j]
                accumulate_count[labels[j]] += 1
                count+=1

        if count >= BATCH_SIZE_ADAPT :
            for key in range(0,6):
                #print(key)
                X_target = X_target + sample_per_class[key]
                y_target = np.concatenate((y_target,np.full(len(sample_per_class[key]), key,dtype=int))) 
            #print(Counter(y_target))
            X_target = np.array(X_target)
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
                epochs=60,callbacks=[EarlyStoppingByAccuracy()])
            X_target = []
            y_target = []
            count = 0
    
    return results
def get_threshold_gas(X_train_proj, y_train):
    class_count_train = Counter(y_train)

    def def_value_b():
        return np.zeros(X_train_proj[0].shape)
    train_proj_by_class = defaultdict(def_value_b)
    for i in range(1, len(X_train_proj)):
        train_proj_by_class[y_train[i]] += X_train_proj[i]
    for i in range(0,6):
        train_proj_by_class[i] = train_proj_by_class[i] / class_count_train[i]
    distance_train = defaultdict(list)
    avg_distance_train = {}
    for i in range(len(X_train_proj)):
        distance_train[y_train[i]].append(np.linalg.norm(X_train_proj[i] - train_proj_by_class[y_train[i]]))

    for i in range(0,6):
        avg_distance_train[i] = np.mean(distance_train[i])
    return avg_distance_train, train_proj_by_class
def train_gas_online_saved(X_train,y_train,X_test,y_test,verbose = True):
    optimizer3=tf.keras.optimizers.Adam(learning_rate=0.0003 )
    encoder_r = encoder_net()
    projector_z = projector_net()
    
    def supervised_model():
    
        inputs = Input((128,  ))
        encoder_r.trainable = False
        projector_z.trainable = False
        r = projector_z(encoder_r(inputs, training=False), training=False)
        #r = encoder_r(inputs, training=False)
        outputs = Dense(6, activation='softmax')(r)

        supervised_model = Model(inputs, outputs)
        return supervised_model
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-3)
    supervised_classifier = supervised_model()
    print("loading weights")
    supervised_classifier.compile(optimizer=optimizer2,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    encoder_r.load_weights("./SCL_encoder_Adam_10.20.h5")
    projector_z.load_weights("./SCL_projector_Adam_10.20.h5")
    supervised_classifier.load_weights('./Best_projected_350_epochs_10.20.h5')
    
    X_train_proj = projector_z.predict(encoder_r.predict(X_train))
    avg_distance_train, train_proj_by_class = get_threshold_gas(X_train_proj, y_train)

    #test for initial feature accuracy
    batches = [1244, 1586,161,197,2300,3613,294,470,3600]

    EPOCH_FEATURE_ADAPT = 25
    EPOCH_CLASSIFIER_ADAPT = 60
    LOG_EVERY = 10
    BS = 80
    AUTO = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 80

    NUM_TEST = len(X_test)
    BS_ADAPT = 200
    NUM_BATCH = NUM_TEST // BS_ADAPT
    BATCH_SIZE_ADAPT = 20
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


    # Set samples for each class in train
    sample_per_class = [[],[],[],[],[],[]]
    for i in range(len(y_train)):
        sample_per_class[y_train[i]].append(X_train[i])

    results = []
    count = 0
    X_target = []
    y_target = []
    BATH_SIZE_ADAPT = 60
    accumulate_count = [0,0,0,0,0,0]
    
    for i in range(0,NUM_BATCH+1):
        
        X_batch = X_test[i*BS_ADAPT:(i+1)*BS_ADAPT]
        batch_proj = projector_z.predict(encoder_r.predict(X_batch))
        sample_score = supervised_classifier.predict(X_batch)
        labels = sample_score.argmax(axis=1) 
        results = np.concatenate((results, labels))  
    
        
        for j in range(0, len(X_batch)):
            d = np.linalg.norm(batch_proj[j] - train_proj_by_class[labels[j]])
            if d < avg_distance_train[labels[j]] + 0.7:
                #X_target.append(X_batch[j])
                #y_target.append(labels[j])
                if accumulate_count[labels[j]] >= len(sample_per_class[labels[j]]):
                    accumulate_count[labels[j]] = 0
                sample_per_class[labels[j]][accumulate_count[labels[j]]] = X_batch[j]
                accumulate_count[labels[j]] += 1
                count+=1

        if count >= BATCH_SIZE_ADAPT :
            for key in range(0,6):
                #print(key)
                X_target = X_target + sample_per_class[key]
                y_target = np.concatenate((y_target,np.full(len(sample_per_class[key]), key,dtype=int))) 
            #print(Counter(y_target))
            X_target = np.array(X_target)
            X_target = np.asarray(X_target).astype('float32')
            y_target = np.asarray(y_target).astype('int32')
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
                epochs=60,callbacks=[EarlyStoppingByAccuracy()])
            X_target = []
            y_target = []
            count = 0
            

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