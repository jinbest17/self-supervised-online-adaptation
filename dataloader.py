#from sklearn.utils import shuffle
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
def gray_scale(image):
    """
    Convert images to gray scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def local_histo_equalize(image):
    """
    Apply local histogram equalization to grayscale images.
        Parameters:
            image: A grayscale image.
    """
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local
def preprocess(data):
    """
    Applying the preprocessing steps to the input data.
        Parameters:
            data: An np.array compatible with plt.imshow.
    """
    gray_images = list(map(gray_scale, data))
    equalized_images = list(map(local_histo_equalize, gray_images))
    
    return equalized_images

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
        X_train = np.asarray(X_train).astype(np.float32)
        print(type(X_train))
        print(type(X_test))
        le = preprocessing.LabelEncoder()
        y_train = np.asarray(le.fit_transform(y_train)).astype(np.int32)
        y_test = np.asarray(le.transform(y_test)).astype(np.int32)
        
        return X_train, y_train, X_test, y_test
    elif dataset_name == 'mnist':
        mnist = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0

        X_train = np.expand_dims(X_train, axis=-1)
        X_train = np.asarray(X_train).astype(np.float32)
        X_test = np.expand_dims(X_test, axis=-1)
        print(type(X_train))
        print(type(X_test))

        le = preprocessing.LabelEncoder()
        y_train = np.asarray(le.fit_transform(y_train)).astype(np.int32)
        y_test = np.asarray(le.transform(y_test)).astype(np.int32)

        return X_train, y_train, X_test, y_test
    
    elif dataset_name == 'cifar':
        cifar = tf.keras.datasets.cifar10
        (X_train, y_train), (X_test, y_test) = cifar.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0
        
        le = preprocessing.LabelEncoder()
        y_train = np.asarray(le.fit_transform(y_train)).astype(np.int32)
        y_test = np.asarray(le.transform(y_test)).astype(np.int32)
        
        return X_train, y_train, X_test, y_test

        
    elif dataset_name == 'gtsrb':
        training_file = "./traffic-signs-data/train.p"
        validation_file= "./traffic-signs-data/valid.p"
        testing_file = "./traffic-signs-data/test.p"

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        #with open(validation_file, mode='rb') as f:
            #valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
            
        # Mapping ClassID to traffic sign names
        signs = []
        with open('signnames.csv', 'r') as csvfile:
            signnames = csv.reader(csvfile, delimiter=',')
            next(signnames,None)
            for row in signnames:
                signs.append(row[1])
            csvfile.close()
        X_train, y_train = train['features'], train['labels']
        X_valid, y_valid = valid['features'], valid['labels']
        X_test, y_test = test['features'], test['labels']
        X_train, y_train = shuffle(X_train, y_train)
        
                
