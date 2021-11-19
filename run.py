import train
import train_MNIST
import dataloader
import numpy as np
import sys
import pickle

def run_gas():
    X_train, y_train, X_test, y_test = dataloader.load_data('gas')
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3 = train.train_gas_offline(X_train, y_train, X_test, y_test)
    results = train.train_gas_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train, y_train,X_test)
    train.evaluate(y_test, results)
    
def run_mnist():
    X_train, y_train, X_test, y_test = dataloader.load_data('mnist')
    # noise
    noise = np.random.normal(0, 0.3, X_test.shape)
    X_test_noise = X_test + noise
    #print(X_test_noise.shape)
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)
    result_noise = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_noise)
    # rotate
    X_test_rotate = np.array([np.rot90(img) for img in X_test])
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)
    result_rotate = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_rotate)

    # permutation
    X_test_perm_interim = np.array([np.random.permutation(img) for img in X_test])
    X_test_perm_rot = np.array([np.rot90(img) for img in X_test_perm_interim])
    X_test_perm = np.array([np.random.permutation(img) for img in X_test_perm_rot])
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)
    result_perm = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_perm)
    
    results = {
        'results_noise' :results_noise,
        'results_rotate':results_rotate,
        'results_perm':results_perm,
        'true_label':y_test
    }
    pickle.dump(results, open('./mnist_results.p', "wb"))

#def run_gas_baseline():
    #X_train, y_train, X_test, y_test = dataloader.load_data('gas')
    #results = D3.train(X_train, y_train, X_test, y_test)
    #D3.evaluate(results)
    
args = sys.argv
dataset_name = args[1]
if len(args) == 3:
    if args[2] == 'd3':
        run_gas_baseline()
    elif args[2] == 'lssvm':
        print('lssvm')
else:
    if dataset_name == 'gas':
        run_gas()
    elif dataset_name == 'mnist':
        run_mnist()
    