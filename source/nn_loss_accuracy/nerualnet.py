# Gradient Descent
import numpy as np

# 輸入每層神經元數目的陣列，例如 shape_list = [784, 100, 10]
def make_param(shape_list):
    w_list = []
    b_list = []
    
    for i in range(len(shape_list) - 1):
        w = np.random.randn(shape_list[i], shape_list[i + 1])
        b = np.ones(shape_list[i + 1]) / 10
        w_list.append(w)
        b_list.append(b)
    return w_list, b_list # w[0] = w', w[1] = w'', b[0] = b', b[1] = b''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inner_product(x, w, b):
    return np.dot(x, w) + b

def activation_func(x, w, b):
    return sigmoid(inner_product(x, w, b))

def calculate(x, w_list, b_list, y):  # ++
    val_dict = {}
    a1 = inner_product(x, w_list[0], b_list[0])
    y1 = sigmoid(a1)
    
    a2 = inner_product(y1, w_list[1], b_list[1])
    y2 = sigmoid(a2)
    
    y2 /= np.sum(y2, axis = 1, keepdims=True)
    
    S = 1/(2*len(y2))*(y2 - y)**2 # ++
    L = np.sum(S) # ++
    val_dict['y1'] = y1
    val_dict['y2'] = y2
    val_dict['loss'] = L # ++
    return val_dict

def update(x_train, w_list, b_list, y_train, lr):
    val_dict = calculate(x_train, w_list, b_list, y_train) # ++
    y1, y2 = val_dict['y1'], val_dict['y2']
    d12_d11 = 1.0
    d11_d9 = 1/x_train.shape[0]*(y2 - y_train)
    d9_d8 = y2*(1-y2)
    d8_d7 = 1
    d8_d6 = y1.T
    d8_d5 = w_list[1].T
    d5_d4 = y1*(1-y1)
    d4_d3 = 1
    d4_d2 = x_train.T
    
    d12_d8 = d12_d11 * d11_d9 * d9_d8
    b_list[1] -= lr * np.sum(d12_d8 * d8_d7, axis = 0)
    w_list[1] -= lr * np.dot(d8_d6, d12_d8)
    
    #d12_d8 = d12_d11 * d11_d9 * d9_d8
    d12_d5 = np.dot(d12_d8, d8_d5)
    d12_d4 = d12_d5 * d5_d4
    b_list[0] -= lr * np.sum(d12_d4 * d4_d3, axis = 0)
    w_list[0] -= lr * np.dot(d4_d2, d12_d4)
    
    return w_list, b_list
#
def accuracy(x, w_list, b_list, y):
    val_dict = calculate(x, w_list, b_list, y)
    predict = val_dict['y2']
    result = np.where(np.argmax(y, axis=1) == np.argmax(predict, axis=1), 1, 0).mean()
    return result

def loss(x, w_list, b_list, y):
    val_dict = calculate(x, w_list, b_list, y)
    return val_dict['loss']
#
