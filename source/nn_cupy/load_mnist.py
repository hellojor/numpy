# download data base at http://yann.lecun.com/exdb/mnist/
import cupy as cp 
import numpy as np
import gzip

key_file = {
'x_train':'../train-images-idx3-ubyte.gz',
'y_train':'../train-labels-idx1-ubyte.gz',
'x_test':'../t10k-images-idx3-ubyte.gz',
'y_test':'../t10k-labels-idx1-ubyte.gz'
}

def load_image(filename):
    with gzip.open(filename, 'rb') as f:
        imgs = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        imgs = cp.asarray(imgs)
    return imgs   

def load_label(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        labels = cp.asarray(labels)
        one_hot_labels = cp.zeros((labels.shape[0], 10))
        for i in range(labels.shape[0]):
            one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

def convert_into_numpy(key_file):
    dataset = {}

    dataset['x_train'] = load_image(key_file['x_train'])
    dataset['y_train'] = load_label(key_file['y_train'])
    dataset['x_test'] = load_image(key_file['x_test'])
    dataset['y_test'] = load_label(key_file['y_test'])
    
    return dataset


def load_mnist():

    dataset = convert_into_numpy(key_file)

    dataset['x_train'] = dataset['x_train'].astype(np.float32)
    dataset['x_test'] = dataset['x_test'].astype(np.float32)
    dataset['x_train'] /= 255.0
    dataset['x_test'] /= 255.0

    dataset['x_train'] = dataset['x_train'].reshape(-1, 28*28)
    dataset['x_test'] = dataset['x_test'].reshape(-1, 28*28)
    return dataset