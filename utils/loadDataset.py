import h5py
import numpy as np

from sklearn.utils import shuffle


def load_train_dataset(path):
    h5f = h5py.File(path, 'r')

    train_x = h5f.get('x_train_dataset')
    train_x = np.float32(np.array(train_x))
    train_y = h5f.get('y_train_dataset')
    train_y = np.float32(np.array(train_y))

    return train_x, train_y


def load_test_dataset(path):
    h5f = h5py.File(path, 'r')

    test_x = h5f.get('x_test_dataset')
    test_x = np.array(test_x)
    test_y = h5f.get('y_test_dataset')
    test_y = np.array(test_y)

    return test_x, test_y


def load_predict_dataset(path, i):
    i = str(i)
    h5f = h5py.File(path, 'r')
    predict_x = h5f.get('x_predict_dataset_' + i)
    predict_x = np.array(predict_x)

    predict_y = h5f.get('y_predict_dataset_' + i)
    predict_y = np.array(predict_y)

    return predict_x, predict_y


def load_unique_predict_dataset(path):
    h5f = h5py.File(path, 'r')
    predict_x = h5f.get('x_predict_dataset')
    predict_x = np.array(predict_x)

    predict_y = h5f.get('y_predict_dataset')
    predict_y = np.array(predict_y)

    return predict_x, predict_y


def load_repredict_dataset():
    h5f = h5py.File('dataset.h5', 'r')
    predict_x = h5f.get('repredict_x')
    predict_x = np.array(predict_x)

    predict_y = h5f.get('repredict_y')
    predict_y = np.array(predict_y)

    return predict_x, predict_y
