import train
import train_MNIST
import dataloader
import numpy as np
import sys


def run_gas():
    X_train, y_train, X_test, y_test = dataloader.load_data('gas')
    results = train.train_gas(X_train, y_train, X_test, y_test)
    train.evaluate(y_test, results)
    
def run_mnist():
    X_train, y_train, X_test, y_test = dataloader.load_data('mnist')
    # noise
    noise = np.random.normal(0, 0.3, X_test.shape)
    X_test_noise = X_test + noise
    #print(X_test_noise.shape)
    results_noise = train_MNIST.train_mnist(X_train, y_train, X_test_noise, y_test)
    # rotate
    X_test_rotate = np.array([np.rot90(img) for img in X_test])
    results_rotate = train_MNIST.train_mnist(X_train, y_train, X_test_rotate, y_test)

    # permutation
    X_test_perm_interim = np.array([np.random.permutation(img) for img in X_test])
    X_test_perm_rot = np.array([np.rot90(img) for img in X_test_perm_interim])
    X_test_perm = np.array([np.random.permutation(img) for img in X_test_perm_rot])
    results_perm = train_MNIST.train_mnist(X_train, y_train, X_test_perm, y_test)
    
    results = {
        'results_noise' :results_noise,
        'results_rotate':results_rotate,
        'results_perm':results_perm,
        'true_label':y_test
    }
    pickle.dump(results, open('./mnist_results.p', "wb"))
    
args = sys.argv
dataset_name = args[1]

if dataset_name == 'gas':
    run_gas()
elif dataset_name == 'mnist':
    run_mnist()
    