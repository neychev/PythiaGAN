import sys

if len(sys.argv) != 7:
    raise ValueError()

params = sys.argv[1:]
GPU_IDX = params[0]
n_steps = int(params[1])
n_initial_points = int(params[2])
n_epochs = int(params[3])
batch_size = int(params[4])
alpha0 = float(params[5])

import time
import os
os.environ['OMP_NUM_THREADS'] = '1'
### configuring keras to use fixed part of GPU
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.095
config.gpu_options.visible_device_list = GPU_IDX
set_session(tf.Session(config=config))

import keras
from keras import backend as K
from keras.models import model_from_json

import json
import pickle

import numpy as np
from skimage import transform

from skopt import Optimizer
from skopt.space import Real

### Loading dataset
from mnist import load_dataset


def rotate_dataset(dataset, angle):
    return np.array(list(map(lambda x: transform.rotate(x, angle), dataset)))


def get_base_data(half_n_samples, alpha0=0):
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    all_data = np.vstack((X_train, X_val, X_test))
    all_data = all_data.reshape(-1, 28, 28)
    subsample_indices = np.random.choice(len(all_data), half_n_samples, replace=False)

    all_data = all_data[subsample_indices]
    
    if alpha0 != 0:
        all_data = rotate_dataset(all_data, alpha0)
    return all_data


def get_X_and_y(base_dataset, target_dataset, angle):
    rotated_datased = rotate_dataset(base_dataset, angle)
    whole_dataset = np.vstack((target_dataset, rotated_datased)).reshape(-1, 784)
    shuffled_indices = np.arange(len(whole_dataset))
    np.random.shuffle(shuffled_indices)
    
    whole_dataset = whole_dataset[shuffled_indices]
    
    _labels = np.zeros(2*base_dataset.shape[0], dtype=bool)
    _labels[base_dataset.shape[0]:] = True
    _labels = _labels[:, None]
    
    whole_labels = np.array(np.hstack((_labels, ~_labels)), dtype=int)
    whole_labels = whole_labels[shuffled_indices]
    
    return whole_dataset, whole_labels


def load_model(path_to_model):
    with open(path_to_model, 'r') as iofile:
        model_architecture = json.load(iofile)    
    model = model_from_json(model_architecture)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
    return


class myOptimizer:
    def __init__(self, 
                 dataset_size, 
                 alpha0=0,
                 n_epochs=5, 
                 batch_size=64, 
                 path_to_model='model_architecture.json'):
        self._dataset_size = dataset_size
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._alpha0 = alpha0
        self.base_data = self._read_data()
        self.target_data = self._get_rotated_data()
        self.results_dict = dict()
        self.model = load_model(path_to_model)
    
    def _read_data(self):
        return get_base_data(self._dataset_size)
    
    def _get_rotated_data(self):
        return rotate_dataset(self.base_data, self._alpha0)
    
    def target_function(self, params):
        alpha = params[0]

        reset_weights(self.model)
    #     print('here')
        X, y = get_X_and_y(self.base_data, self.target_data, alpha)
        val_border = -int(np.floor(X.shape[0]/20))
        test_border = -int(np.floor(X.shape[0]/5))
        X_val, y_val = X[val_border:], y[val_border:]
        X_test, y_test = X[test_border:val_border], y[test_border:val_border]
        X, y = X[:test_border], y[:test_border]

        history = self.model.fit(X, y,
                            batch_size=self._batch_size,
                            epochs=self._n_epochs,
                            verbose=0,
                            validation_data=(X_val, y_val))
        _logloss, _accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        target_score = np.log(2) - _logloss

        self.results_dict[alpha] = {
            'score': target_score,
            'model_weights': self.model.get_weights()
        }

        return target_score
    

if __name__ == '__main__':
    dimensions = [Real(name='alpha', low=-180.0, high=180.0)]
    
    for dataset_size in [2000, 3000, 5000, 10000, 20000, 35000, 70000]:
        s_time = time.time()
        opt = Optimizer(dimensions=dimensions, n_initial_points=n_initial_points)
        optimization_attempt = myOptimizer(dataset_size=dataset_size, 
                                           alpha0=alpha0,
                                           n_epochs=n_epochs, 
                                           batch_size=batch_size)
        p_time = time.time()
        res = opt.run(optimization_attempt.target_function, n_iter=n_steps)
        e_time = time.time()
        
        with open('{}_{}_start.pcl'.format(alpha0, dataset_size), 'wb') as iofile:
            save_dict = {
                'opt': opt,
                'alpha0': alpha0,
                'dataset_size': dataset_size,
                'results_dict': optimization_attempt.results_dict,
                'raw_sysargv_params': params,
                'time': (s_time, p_time, e_time)
            }
            pickle.dump(save_dict, iofile)
        
        print('Process with alpha0 = {} finished with dataset size {}'.format(alpha0, dataset_size))
