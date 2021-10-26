#開始學習
    
import numpy as np
import neuralnet_acc_loss as nl
import load_mnist_acc_loss

dataset = load_mnist_acc_loss.load_mnist()
x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']

w_list, b_list = nl.make_params([784, 100, 10])

train_time = 10000
batch_size = 1000

# 建立陣列儲存準確率與損失的變動值
total_acc_list = []
total_loss_list = []


for epoch in range(3):

    ra = np.random.randint(60000,size=60000)

    for i in range(60):
        x_batch = x_train[ra[i*1000:(i+1)*1000],:]
        y_batch = y_train[ra[i*1000:(i+1)*1000],:]
        w_list, b_list = nl.update(x_batch, w_list, b_list, y_batch, eta=2.0)  
# 這裡進行參數的更新。eta 為學習率，決定參數更新時的倍率。
# 這裡設定學習率為 2.0
# 實際上這個值會經由嘗試錯誤中來決定


     
    acc_list = []
    loss_list = []
        
    for k in range(10000//1000):
        
        x_batch, y_batch = x_test[k*batch_size:(k+1)*batch_size, :], y_test[k*batch_size:(k+1)*batch_size, :]
        
        acc_val = nl.accuracy(x_batch, w_list, b_list, y_batch)
        loss_val = nl.loss(x_batch, w_list, b_list, y_batch)
        
        acc_list.append(acc_val)
        loss_list.append(loss_val)

    # 計算平均的準確率      
    acc = np.mean(acc_list)
    loss = np.mean(loss_list)
    
    total_acc_list.append(acc)
    total_loss_list.append(loss)
    print("epoch:%d, Accuracy: %f, Loss: %f"%(epoch, acc, loss))
    
   
        
import matplotlib.pyplot as plt
plt.subplot(211)
plt.plot(np.arange(0, len(total_acc_list)),total_acc_list)
plt.title('accuracy')
plt.subplot(212)
plt.plot(np.arange(0, len(total_acc_list)), total_loss_list)
plt.title('loss')
plt.tight_layout()
plt.show()









