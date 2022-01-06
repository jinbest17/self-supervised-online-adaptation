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
	projector = tf.keras.models.Sequential([
		Dense(80, activation="relu"),
		UnitNormLayer()
	])

	return projector
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
    CONFIDENCE_THRESHOLD = 0.85
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
        
    # Calculate score for each class in train
    score_batch1 = supervised_classifier.predict(X_train)
    score_max_1 = score_batch1.max(axis=1)
    #scores_per_class = defaultdict(list)
    scores_per_class = [0,0,0,0,0,0]
    mean_score = [0,0,0,0,0,0]
    for i in range(len(y_train)):
        
        scores_per_class[y_train[i]]+=score_max_1[i]
    mean_score = [scores_per_class[key] / 98 for key in range(0,6)]
    #mean_score = {key:np.mean(scores_per_class[key]) for key in scores_per_class}
    if verbose == True:
        print("mean softmax score for training samples in each class: ",mean_score)

    
    
    results = []
    results = []
    max_scores = []
    pseudo_label_accuracy = []
    contrastive_loss = []
    training_accuracy = []
    retrains=[0]
    label_accuracy = 0
    count = 0
    BATH_SIZE_ADAPT = 60
    new_samples_dict = defaultdict(list)
    accumulate_count = [0,0,0,0,0,0]
    #lowerbound = [0,0,0,0,0,0]
    lowerbound = [mean_score[i] * CONFIDENCE_THRESHOLD for i in range(0,6)]
    
    for i in range(0,len(X_test)):
        
        sample_score = supervised_classifier.predict(X_test[i].reshape(1,128))
        label = sample_score.argmax() 
        results.append(label)    
        max_score = sample_score.max()
        max_scores.append(max_score)
        if max_score < mean_score[label]  and  max_score > lowerbound[label]:
            # add to new training sample
            if accumulate_count[label] >= len(sample_per_class[label]):
                accumulate_count[label] = 0
            sample_per_class[label][accumulate_count[label]] = X_test[i]
            accumulate_count[label] += 1
            count+=1
            if label == y_test[i]:
                label_accuracy += 1
            
            
        # if we accumulate enough samples
        if count >= BATH_SIZE_ADAPT:
            pseudo_label_accuracy.append(label_accuracy/60)
            label_accuracy = 0
            retrains.append(i)
            if verbose == True:
                print("New adaptation initiated at sample", i)
                print_batch_size(sample_per_class)
            # form target batch
            X_target =[]
            y_target = np.array([],dtype=int)

            for key in range(0,6):
                #print(key)
                X_target = X_target + sample_per_class[key]
                y_target = np.concatenate((y_target,np.full(98, key,dtype=int)))        
            
            X_target = np.array(X_target)
            
            #print(X_target.shape, y_target.shape)
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
                    if verbose == True:
                        print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
                        print("Encoder train exiting")
                    break
                if verbose == True and (epoch % LOG_EVERY) == 0:
                    print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
            encoder_r.trainable = False
            projector_z.trainable = False
            contrastive_loss.append(train_loss_results[-1])
            hist=supervised_classifier.fit(train_ds,
                epochs=EPOCH_CLASSIFIER_ADAPT, verbose=1 if verbose else 0,callbacks=[EarlyStoppingByAccuracy()])
            training_accuracy.append(hist.history['loss'])
            count = 0
            
            
            # Calculate score for each class in train
            score_batch1 = supervised_classifier.predict(X_target)
            score_max_1 = score_batch1.max(axis=1)
            #scores_per_class = defaultdict(list)
            
            scores_per_class = [0,0,0,0,0,0]
            for j in range(len(y_target)):
                scores_per_class[y_target[j]]+=score_max_1[j]
            mean_score = [scores_per_class[key] / 98 for key in range(0,6)]
            #mean_score = {key:np.mean(scores_per_class[key]) for key in scores_per_class}
            if verbose == True:
                print("mean softmax score for training samples in each class: ",mean_score)
            
            
            
            #optimizer3=tf.keras.optimizers.Adam(learning_rate=0.0003 )
            #optimizer2=tf.keras.optimizers.Adam(learning_rate=1e-3 )

            lowerbound = [mean_score[j] * CONFIDENCE_THRESHOLD for j in range(0,6)]
    saved = {
      'contrastive_loss':contrastive_loss,
      'results':results,
      'steps': retrains,
      'scores':max_scores,
      'pseudo_label_accuracy':pseudo_label_accuracy,
      'y_test':y_test
    }
    pickle.dump(saved, open('saved2.p','wb'))
    return results

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

    supervised_classifier.compile(optimizer=optimizer2,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    encoder_r.load_weights("./SCL_encoder_Adam_10.20.h5")
    projector_z.load_weights("./SCL_projector_Adam_10.20.h5")
    supervised_classifier.load_weights('./Best_projected_350_epochs_10.20.h5')
    
    #test for initial feature accuracy
    batches = [1244, 1586,161,197,2300,3613,294,470,3600]
    count = 0
    for i in range(len(batches)):
        supervised_classifier.evaluate(X_test[count:count+batches[i]], y_test[count:count+batches[i]])
        count+=batches[i]
    EPOCH_FEATURE_ADAPT = 25
    EPOCH_CLASSIFIER_ADAPT = 50
    LOG_EVERY = 10
    BS = 80
    AUTO = tf.data.experimental.AUTOTUNE
    CONFIDENCE_THRESHOLD = 0.8
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
        
    # Calculate score for each class in train
    score_batch1 = supervised_classifier.predict(X_train)
    score_max_1 = score_batch1.max(axis=1)
    #scores_per_class = defaultdict(list)
    scores_per_class = [0,0,0,0,0,0]
    mean_score = [0,0,0,0,0,0]
    for i in range(len(y_train)):
        
        scores_per_class[y_train[i]]+=score_max_1[i]
    mean_score = [scores_per_class[key] / 98 for key in range(0,6)]
    #mean_score = {key:np.mean(scores_per_class[key]) for key in scores_per_class}
    if verbose == True:
        print("mean softmax score for training samples in each class: ",mean_score)

    
    
    results = []
    results = []
    #max_scores = []

    count = 0
    BATH_SIZE_ADAPT = 60
    new_samples_dict = defaultdict(list)
    accumulate_count = [0,0,0,0,0,0]
    #lowerbound = [0,0,0,0,0,0]
    lowerbound = [mean_score[i] * CONFIDENCE_THRESHOLD for i in range(0,6)]
    
    for i in range(0,len(X_test)):
        
        sample_score = supervised_classifier.predict(X_test[i].reshape(1,128))
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
            if verbose == True:
                print("New adaptation initiated at sample", i)
                print_batch_size(sample_per_class)
            # form target batch
            X_target =[]
            y_target = np.array([],dtype=int)

            for key in range(0,6):
                #print(key)
                X_target = X_target + sample_per_class[key]
                y_target = np.concatenate((y_target,np.full(98, key,dtype=int)))        
            
            X_target = np.array(X_target)
            
            #print(X_target.shape, y_target.shape)
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
                    if verbose == True:
                        print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
                        print("Encoder train exiting")
                    break
                if verbose == True and (epoch % LOG_EVERY) == 0:
                    print("Epoch: {} Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))
            encoder_r.trainable = False
            projector_z.trainable = False
            
            supervised_classifier.fit(train_ds,
                epochs=EPOCH_CLASSIFIER_ADAPT, verbose=1 if verbose else 0,callbacks=[EarlyStoppingByAccuracy()])
            
            count = 0
            
            
            # Calculate score for each class in train
            score_batch1 = supervised_classifier.predict(X_target)
            score_max_1 = score_batch1.max(axis=1)
            #scores_per_class = defaultdict(list)
            
            scores_per_class = [0,0,0,0,0,0]
            for j in range(len(y_target)):
                scores_per_class[y_target[j]]+=score_max_1[j]
            mean_score = [scores_per_class[key] / 98 for key in range(0,6)]
            #mean_score = {key:np.mean(scores_per_class[key]) for key in scores_per_class}
            if verbose == True:
                print("mean softmax score for training samples in each class: ",mean_score)
            
            
            
            #optimizer3=tf.keras.optimizers.Adam(learning_rate=0.0003 )
            #optimizer2=tf.keras.optimizers.Adam(learning_rate=1e-3 )

            lowerbound = [mean_score[j] * CONFIDENCE_THRESHOLD for j in range(0,6)]
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