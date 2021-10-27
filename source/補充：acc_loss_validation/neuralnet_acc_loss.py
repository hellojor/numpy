#---------梯度下降法，定義神經元的部分---------
import numpy as np

# 輸入每層神經元數目的陣列，例如 shape_list = [784, 100, 10]
def make_params(shape_list):
    w_list = []
    b_list = []
    for i in range(len(shape_list)-1):
        
        # 產生初始值為遵從標準常態分佈的亂數
        weight = np.random.randn(shape_list[i], shape_list[i+1])
        
        # 始值全部設定 0.1
        bias = np.ones(shape_list[i+1])/10.0
    
        w_list.append(weight)
        b_list.append(bias)
    
    return w_list, b_list


def sigmoid(x): # sigmoid 函式
    return 1/(1+np.exp(-x))

def inner_product(x_train, w, b): # 內積再加上偏權值
    return np.dot(x_train, w)+ b

def activation(x_train, w, b):
    return sigmoid(inner_product(x_train, w, b))


# 以配列的型別返回每層的計算結果
def calculate(x_train, w_list, b_list, y_train):
    
    val_list = {}
    a_1 = inner_product(x_train, w_list[0], b_list[0]) # (N, 1000)
    y_1 = sigmoid(a_1) # (N, 1000)
    a_2 = inner_product(y_1, w_list[1], b_list[1]) # (N, 10)
    # 這是原本要取得的值。(N,10)
    y_2 = sigmoid(a_2)
    # 這裡進行簡單的正規化
    y_2 /= np.sum(y_2, axis=1, keepdims=True)
    S = 1/(2*len(y_2))*(y_2 - y_train)**2
    L = np.sum(S)
    val_list['a_1'] = a_1
    val_list['y_1'] = y_1
    val_list['a_2'] = a_2
    val_list['y_2'] = y_2
    val_list['S'] = S
    val_list['L'] = L
    return val_list


def update(x_train, w_list, b_list, y_train, eta):
    val_list = {}
    val_list = calculate(x_train, w_list, b_list, y_train)
#   a_1 = val_list['a_1']
    y_1 = val_list['y_1']
#   a_2 = val_list['a_2']
    y_2 = val_list['y_2']
#   S = val_list['S']
#   L = val_list['L']

    d12_d11 = 1.0
    d11_d9 = 1/x_train.shape[0]*(y_2 - y_train)
    d9_d8 = y_2*(1.0 - y_2)
    d8_d7 = 1.0
    d8_d6 = np.transpose(y_1)
    d8_d5 = np.transpose(w_list[1])
    d5_d4 = y_1 * (1 - y_1)
    d4_d3 = 1.0
    d4_d2 = np.transpose(x_train)



    #以下計算 d12_d7_&_d12_d6
    d12_d8 = d12_d11 * d11_d9 * d9_d8
    b_list[1] -= eta*np.sum(d12_d8 * d8_d7, axis=0)
    w_list[1] -= eta*np.dot(d8_d6, d12_d8)
    
    #以下計算 d12_d3_&_d12_d2
    d12_d8 = d12_d11 * d11_d9 * d9_d8
    d12_d5 = np.dot(d12_d8, d8_d5)
    d12_d4 = d12_d5 * d5_d4
    
    b_list[0] -= eta * np.sum(d12_d4 * d4_d3, axis=0)
    w_list[0] -= eta * np.dot(d4_d2, d12_d4)
    
    return w_list, b_list




# 這裡進行預測
def predict(X, w_list, b_list, t):
    val_list = calculate(X, w_list, b_list, t)
    y_2 = val_list['y_2']
    result = np.zeros_like(y_2)
    # 對應樣本數
    for i in range(y_2.shape[0]):
        result[i, np.argmax(y_2[i])] = 1
    return result


# 這裡計算準確率    
def accuracy(X, w_list, b_list, t):
    pre = predict(X, w_list, b_list, t)
    result = np.where(np.argmax(t, axis=1)==np.argmax(pre, axis=1), 1, 0)
    acc = np.mean(result)
    return acc

# 這裡計算總損失
def loss(X, w_list, b_list, t):
    L = calculate(X, w_list, b_list, t)['L']
    return L


