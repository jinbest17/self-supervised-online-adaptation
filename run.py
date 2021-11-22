import train
import train_MNIST
import train_baseline_lssvm
import train_baseline_ot
import train_mnist_ot
import dataloader
import numpy as np
import sys
import pickle

def run_gas():
    X_train, y_train, X_test, y_test = dataloader.load_data('gas')
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3 = train.train_gas_offline(X_train, y_train, X_test, y_test)
    results = train.train_gas_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train, y_train,X_test)
    train.evaluate(y_test, results)
def run_gas_from_saved():
    X_train, y_train, X_test, y_test = dataloader.load_data('gas')
    results = train.train_gas_online_saved(X_train, y_train,X_test,y_test)
    train.evaluate(y_test, results)

    
def run_mnist_rotate():
    X_train, y_train, X_test, y_test = dataloader.load_data('mnist')
   
    # rotate
    X_test_rotate = np.array([np.rot90(img) for img in X_test])
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)
    result_rotate = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_rotate)


    results = {
        'results_rotate':result_rotate,
        'true_label':y_test
    }
    pickle.dump(results, open('./mnist_results_rotate.p', "wb"))
def run_mnist_noise():
    X_train, y_train, X_test, y_test = dataloader.load_data('mnist')
    # noise
    noise = np.random.normal(0, 0.3, X_test.shape)
    X_test_noise = X_test + noise
    #print(X_test_noise.shape)
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)
    result_noise = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_noise)
    results = {
        'results_noise' :result_noise,
        'true_label':y_test
    }
    pickle.dump(results, open('./mnist_results_noise.p', "wb"))
def run_mnist_perm():
    X_train, y_train, X_test, y_test = dataloader.load_data('mnist')
    # permutation
    X_test_perm_interim = np.array([np.random.permutation(img) for img in X_test])
    X_test_perm_rot = np.array([np.rot90(img) for img in X_test_perm_interim])
    X_test_perm = np.array([np.random.permutation(img) for img in X_test_perm_rot])
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)
    result_perm = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_perm)
    
    results = {
        'results_perm':result_perm,
        'true_label':y_test
    }
    pickle.dump(results, open('./mnist_results_perm.p', "wb"))
    
def run_mnist_from_saved(transformation):
    N_DATA_TRAIN = 800
    X_train, y_train, X_test, y_test = dataloader.load_data('mnist')
    
    n_train = X_train.shape[0]
    shuffle_idx = np.arange(n_train)
    np.random.shuffle(shuffle_idx)
    result = []
    X_train_small = X_train[shuffle_idx][:N_DATA_TRAIN]
    y_train_small = y_train[shuffle_idx][:N_DATA_TRAIN]
    print(X_train_small.shape, y_train_small.shape)
    if transformation == 'noise':
        # noise
        noise = np.random.normal(0, 0.3, X_test.shape)
        X_test_noise = X_test + noise
        #print(X_test_noise.shape)
        supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3 = train_MNIST.load_model()
        result = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_noise)
    elif transformation == 'rotate':
        # rotate
        X_test_rotate = np.array([np.rot90(img) for img in X_test])
        supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)
        result= train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_rotate)
    elif transformation == 'permutation':
        # permutation
        X_test_perm_interim = np.array([np.random.permutation(img) for img in X_test])
        X_test_perm_rot = np.array([np.rot90(img) for img in X_test_perm_interim])
        X_test_perm = np.array([np.random.permutation(img) for img in X_test_perm_rot])
        supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)
        result = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_perm)
    
    results = {
        'results_'+transformation :result,
        'true_label':y_test
    }
    pickle.dump(results, open('./mnist_results'+transformation+'.p', "wb"))

def run_gas_baseline_ot():
    X_train, y_train, X_test, y_test = dataloader.load_data('gas')    
    model = train_baseline_ot.train_baseline_ot_offline(X_train, y_train)
    results = train_baseline_ot.train_baseline_ot_online(X_train, y_train, X_test, model)
    train_baseline_ot.evaluate(y_test, results)
def run_mnist_ot_from_saved(transformation):
    N_DATA_TRAIN = 800
    X_train, y_train, X_test, y_test = dataloader.load_data('mnist')
    X_test = X_test[:3200]
    n_train = X_train.shape[0]
    shuffle_idx = np.arange(n_train)
    np.random.shuffle(shuffle_idx)
    result = []
    X_train_small = X_train[shuffle_idx][:N_DATA_TRAIN]
    y_train_small = y_train[shuffle_idx][:N_DATA_TRAIN]
    print(X_train_small.shape, y_train_small.shape)
    if transformation == 'noise':
        # noise
        noise = np.random.normal(0, 0.3, X_test.shape)
        X_test_noise = X_test + noise
        #print(X_test_noise.shape)
        supervised_classifier  = train_mnist_ot.load_model()
        result = train_mnist_ot.train_mnist_ot_online( X_train_small, y_train_small, X_test_noise,supervised_classifier)
    elif transformation == 'rotate':
        # rotate
        X_test_rotate = np.array([np.rot90(img) for img in X_test])
        supervised_classifier  = train_mnist_ot.load_model()
        result = train_mnist_ot.train_mnist_ot_online( X_train_small, y_train_small, X_test_rotate,supervised_classifier)
    elif transformation == 'permutation':
        # permutation
        X_test_perm_interim = np.array([np.random.permutation(img) for img in X_test])
        X_test_perm_rot = np.array([np.rot90(img) for img in X_test_perm_interim])
        X_test_perm = np.array([np.random.permutation(img) for img in X_test_perm_rot])
        supervised_classifier  = train_mnist_ot.load_model()
        result = train_mnist_ot.train_mnist_ot_online( X_train_small, y_train_small, X_test_perm,supervised_classifier)
    count = 0
    for i in range(len(result)):
        if result[i] == y_test[i]:
            count+=1
    print(count /len(result) )
    results = {
        'results_'+transformation :result,
        'true_label':y_test
    }
    pickle.dump(results, open('./mnist_results'+transformation+'.p', "wb"))
    
    
def run_gas_baseline_lssvm():
    X_train, y_train, X_test, y_test = dataloader.load_data('gas')    
    model = train_baseline_lssvm.train_baseline_lssvm_offline(X_train, y_train)
    results = train_baseline_lssvm.train_baseline_lssvm_online( X_test, model)
    
    train_baseline_lssvm.evaluate(y_test, results)

    
args = sys.argv
dataset_name = args[1]


if dataset_name == 'gas':
    if len(args) == 3:
        if args[2] == 'ot':
            run_gas_baseline_ot()
        elif args[2] == 'lssvm':
            run_gas_baseline_lssvm()
        elif args[2] == 'SOA':
            run_gas()
    else:

        run_gas_from_saved()
elif dataset_name == 'mnist':
    if len(args) == 4:
        if args[3] == 'noise':
            run_mnist_noise()
        elif args[2] == 'rotate':
            run_mnist_rotate()
        elif args[2] == 'permutation':
            run_mnist_perm()
    else:
        if args[2] == 'ot':
            run_mnist_ot_from_saved(args[3])

        elif args[2] == 'SOA':
            run_mnist_from_saved(args[3])

    