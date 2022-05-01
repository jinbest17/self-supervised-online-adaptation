import train
import train_MNIST
import train_baseline_lssvm
import train_baseline_ot
import train_mnist_ot
import train_CIFAR
import train_GTSRB
import dataloader
import numpy as np
import sys
import pickle
import time
import socket
from collections import Counter
from collections import defaultdict

UDP_IP = "169.254.200.28"
UDP_PORT = 30000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def run_gas():
    X_train, y_train, X_test, y_test = dataloader.load_data('gas')
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3 = train.train_gas_offline(X_train, y_train, X_test, y_test)

    sock.sendto(b's,gas_SOA', (UDP_IP, UDP_PORT))
    st = time.time()
    results = train.train_gas_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train, y_train,X_test,y_test)
    ed = time.time()
    sock.sendto(b't,', (UDP_IP, UDP_PORT))
    with open('time_log.txt', 'a+') as f:
        f.write('gas SOA online exec time: {} secs\n'.format(ed - st))

    train.evaluate(y_test, results)


def run_gas_from_saved():
    X_train, y_train, X_test, y_test = dataloader.load_data('gas')
    sock.sendto(b's,gas_SOA', (UDP_IP, UDP_PORT))
    st = time.time()
    results = train.train_gas_online_saved(X_train, y_train,X_test,y_test)
    ed = time.time()
    sock.sendto(b't,', (UDP_IP, UDP_PORT))
    with open('time_log.txt', 'a+') as f:
        f.write('gas SOA online exec time: {} secs\n'.format(ed - st))

    print('online exec time: {} secs'.format(time.time() - st))

    train.evaluate(y_test, results)

    
def run_mnist_rotate():
    X_train, y_train, X_test, y_test = dataloader.load_data('mnist')
   
    # rotate
    X_test_rotate = np.array([np.rot90(img) for img in X_test])
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)

    sock.sendto(b's,mnist_SOA_rotate', (UDP_IP, UDP_PORT))
    st = time.time()
    result_rotate = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_rotate)
    ed = time.time()
    sock.sendto(b't,', (UDP_IP, UDP_PORT))


    results = {
        'results_rotate':result_rotate,
        'true_label':y_test
    }
    pickle.dump(results, open('./saved_results/mnist_results_rotate.p', "wb"))
def run_mnist_noise():
    X_train, y_train, X_test, y_test = dataloader.load_data('mnist')
    # noise
    noise = np.random.normal(0, 0.3, X_test.shape)
    X_test_noise = X_test + noise
    #print(X_test_noise.shape)
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)
    X_train_proj = projector_z.predict(encoder_r.predict(X_train))
    avg_distance_train, train_proj_by_class = get_threshold_mnist(X_train_proj, y_train)
    sock.sendto(b's,mnist_SOA_noise', (UDP_IP, UDP_PORT))
    st = time.time()
    result_noise = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_noise,train_proj_by_class,avg_distance_train)
    ed = time.time()
    sock.sendto(b't,', (UDP_IP, UDP_PORT))
    with open('time_log.txt', 'a+') as f:
        f.write('mnist SOA noise online exec time: {} secs\n'.format(ed - st))


    results = {
        'results_noise' :result_noise,
        'true_label':y_test
    }
    pickle.dump(results, open('./saved_results/mnist_results_noise.p', "wb"))
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
    pickle.dump(results, open('./saved_results/mnist_results_perm.p', "wb"))
def get_threshold_mnist(X_train_proj, y_train):
    class_count_train = Counter(y_train)

    def def_value_b():
        return np.zeros(X_train_proj[0].shape)
    train_proj_by_class = defaultdict(def_value_b)
    for i in range(1, len(X_train_proj)):
        train_proj_by_class[y_train[i]] += X_train_proj[i]
    for i in range(0,10):
        train_proj_by_class[i] = train_proj_by_class[i] / class_count_train[i]
    distance_train = defaultdict(list)
    avg_distance_train = {}
    for i in range(len(X_train_proj)):
        distance_train[y_train[i]].append(np.linalg.norm(X_train_proj[i] - train_proj_by_class[y_train[i]]))

    for i in range(0,10):
        avg_distance_train[i] = np.mean(distance_train[i])
    return avg_distance_train, train_proj_by_class
   
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
        #supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)
        X_train_proj = projector_z.predict(encoder_r.predict(X_train))
        avg_distance_train, train_proj_by_class = get_threshold_mnist(X_train_proj, y_train)
        sock.sendto(b's,mnist_SOA_noise', (UDP_IP, UDP_PORT))
        st = time.time()
        result = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_noise,train_proj_by_class,avg_distance_train)
        ed = time.time()
        sock.sendto(b't,', (UDP_IP, UDP_PORT))
        with open('time_log.txt', 'a+') as f:
            f.write('mnist SOA noise online exec time: {} secs\n'.format(ed - st))

        
        print('online {} exec time: {} secs'.format(transformation, time.time() - st))

    elif transformation == 'rotate':
        # rotate
        X_test_rotate = np.array([np.rot90(img) for img in X_test])
        supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)

        st = time.time()
        result= train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_rotate)
        print('online {} exec time: {} secs'.format(transformation, time.time() - st))

    elif transformation == 'permutation':
        # permutation
        X_test_perm_interim = np.array([np.random.permutation(img) for img in X_test])
        X_test_perm_rot = np.array([np.rot90(img) for img in X_test_perm_interim])
        X_test_perm = np.array([np.random.permutation(img) for img in X_test_perm_rot])
        supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_MNIST.train_mnist_offline(X_train, y_train)

        st = time.time()
        result = train_MNIST.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_perm)
        print('online {} exec time: {} secs'.format(transformation, time.time() - st))
    
    results = {
        'results_'+transformation :result,
        'true_label':y_test
    }
    pickle.dump(results, open('./saved_results/mnist_results'+transformation+'.p', "wb"))

def run_cifar_rotate():
    X_train, y_train, X_test, y_test = dataloader.load_data('cifar')
   
    # rotate
    X_test_rotate = np.array([np.rot90(img) for img in X_test])
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_CIFAR.train_mnist_offline(X_train, y_train)

    sock.sendto(b's,mnist_SOA_rotate', (UDP_IP, UDP_PORT))
    st = time.time()
    result_rotate = train_CIFAR.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_rotate)
    ed = time.time()
    sock.sendto(b't,', (UDP_IP, UDP_PORT))


    results = {
        'results_rotate':result_rotate,
        'true_label':y_test
    }
    pickle.dump(results, open('./saved_results/cifar_results_rotate.p', "wb"))


def run_cifar_noise():
    X_train, y_train, X_test, y_test = dataloader.load_data('cifar')
    # noise
    noise = np.random.normal(0, 0.3, X_test.shape)
    X_test_noise = X_test + noise
    #print(X_test_noise.shape)
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_CIFAR.train_mnist_offline(X_train, y_train)
    
    sock.sendto(b's,mnist_SOA_noise', (UDP_IP, UDP_PORT))
    st = time.time()
    result_noise = train_CIFAR.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_noise)
    ed = time.time()
    sock.sendto(b't,', (UDP_IP, UDP_PORT))
    with open('time_log.txt', 'a+') as f:
        f.write('mnist SOA noise online exec time: {} secs\n'.format(ed - st))


    results = {
        'results_noise' :result_noise,
        'true_label':y_test
    }
    pickle.dump(results, open('./saved_results/cifar_results_noise.p', "wb"))
def run_cifar_perm():
    X_train, y_train, X_test, y_test = dataloader.load_data('cifar')
    # permutation
    X_test_perm_interim = np.array([np.random.permutation(img) for img in X_test])
    X_test_perm_rot = np.array([np.rot90(img) for img in X_test_perm_interim])
    X_test_perm = np.array([np.random.permutation(img) for img in X_test_perm_rot])
    supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3,X_train_small, y_train_small = train_CIFAR.train_mnist_offline(X_train, y_train)
    result_perm = train_CIFAR.train_mnist_online(supervised_classifier, encoder_r, projector_z, optimizer2, optimizer3, X_train_small, y_train_small, X_test_perm)
    
    results = {
        'results_perm':result_perm,
        'true_label':y_test
    }
    pickle.dump(results, open('./saved_results/cifar_results_perm.p', "wb"))

def run_gas_baseline_ot():
    X_train, y_train, X_test, y_test = dataloader.load_data('gas')    
    model = train_baseline_ot.train_baseline_ot_offline(X_train, y_train)

    sock.sendto(b's,gas_ot', (UDP_IP, UDP_PORT))
    st = time.time()
    results = train_baseline_ot.train_baseline_ot_online(X_train, y_train, X_test, model)
    ed = time.time()
    sock.sendto(b't,', (UDP_IP, UDP_PORT))
    with open('time_log.txt', 'a+') as f:
        f.write('gas ot online exec time: {} secs\n'.format(ed - st))

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

        sock.sendto(b's,mnist_ot', (UDP_IP, UDP_PORT))
        st = time.time()
        result = train_mnist_ot.train_mnist_ot_online( X_train_small, y_train_small, X_test_noise,supervised_classifier)
        ed = time.time()
        sock.sendto(b't,', (UDP_IP, UDP_PORT))
        with open('time_log.txt', 'a+') as f:
            f.write('mnist ot online {} exec time: {} secs'.format(transformation, ed - st))

    elif transformation == 'rotate':
        # rotate
        X_test_rotate = np.array([np.rot90(img) for img in X_test])
        supervised_classifier  = train_mnist_ot.load_model()

        st = time.time()
        result = train_mnist_ot.train_mnist_ot_online( X_train_small, y_train_small, X_test_rotate,supervised_classifier)
        print('online {} exec time: {} secs'.format(transformation, time.time() - st))

    elif transformation == 'permutation':
        # permutation
        X_test_perm_interim = np.array([np.random.permutation(img) for img in X_test])
        X_test_perm_rot = np.array([np.rot90(img) for img in X_test_perm_interim])
        X_test_perm = np.array([np.random.permutation(img) for img in X_test_perm_rot])
        supervised_classifier  = train_mnist_ot.load_model()

        st = time.time()
        result = train_mnist_ot.train_mnist_ot_online( X_train_small, y_train_small, X_test_perm,supervised_classifier)
        print('online {} exec time: {} secs'.format(transformation, time.time() - st))

    count = 0
    for i in range(len(result)):
        if result[i] == y_test[i]:
            count+=1
    print(count /len(result) )

    results = {
        'results_'+transformation :result,
        'true_label':y_test
    }
    pickle.dump(results, open('./saved_results/mnist_results'+transformation+'.p', "wb"))
    
    
def run_gas_baseline_lssvm():
    X_train, y_train, X_test, y_test = dataloader.load_data('gas')    
    model = train_baseline_lssvm.train_baseline_lssvm_offline(X_train, y_train)

    sock.sendto(b's,gas_lssvm', (UDP_IP, UDP_PORT))
    st = time.time()
    results = train_baseline_lssvm.train_baseline_lssvm_online( X_test, model)
    ed = time.time()
    sock.sendto(b't,', (UDP_IP, UDP_PORT))
    with open('time_log.txt', 'a+') as f:
        f.write( 'gas lssvm online exec time: {} secs\n'.format(ed - st))
    
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
    if len(args) == 3:
        if args[2] == 'noise':
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

elif dataset_name == 'cifar':
    if len(args) == 3:
        if args[2] == 'noise':
            run_cifar_noise()
        elif args[2] == 'rotate':
            run_cifar_rotate()
        elif args[2] == 'permutation':
            run_cifar_perm()
    # else:
    #     if args[2] == 'ot':
    #         run_mnist_ot_from_saved(args[3])

    #     elif args[2] == 'SOA':
    #         run_mnist_from_saved(args[3])

    
