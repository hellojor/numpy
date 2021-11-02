import cupy as cp
import nerualnet as nl
import load_mnist as lm

dataset = lm.load_mnist()

x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']
w_list, b_list = nl.make_param([784, 100, 10])

num_epochs = 200
batch_size = 1000
learning_rate = 0.1

total_acc = []
total_loss = []

for epoch in range(num_epochs):
    ra = cp.random.randint(60000, size=60000)
    for i in range(60):
        x_batch = x_train[ra[i*1000:(i+1)*1000], :]
        y_batch = y_train[ra[i*1000:(i+1)*1000], :]
        
        w_list, b_list = nl.update(x_batch, w_list, b_list, y_batch, lr=learning_rate)
    acc_list = []
    loss_list = []
    for k in range(10000//batch_size):
        x_batch, y_batch = x_test[k*batch_size:(k+1)*batch_size], y_test[k*batch_size:(k+1)*batch_size]
        acc = nl.accuracy(x_batch, w_list, b_list, y_batch)
        loss = nl.loss(x_batch, w_list, b_list, y_batch)
        acc_list.append(acc)
        loss_list.append(loss)
    
    acc = cp.mean(cp.asarray(acc_list))
    loss = cp.mean(cp.asarray(loss_list))
    
    total_acc.append(acc)
    total_loss.append(loss)
    print("epoch %4d  Accuracy: %f  Loss: %f" % (epoch, acc, loss))
  
print(y_test[0:10])

val_dict = nl.calculate(x_test, w_list, b_list, y_test)
print(val_dict['y2'][0:10].round(2))
'''
import matplotlib.pyplot as plt
plt.subplot(211)

plt.plot(cp.asnumpy(cp.arange(0, len(total_acc))), cp.asnumpy(total_acc))
plt.title('accuracy')
plt.subplot(212)
plt.plot(cp.asnumpy(cp.arange(0, len(total_acc))), cp.asnumpy(total_loss))
plt.title('loss')
plt.tight_layout()
plt.show()'''