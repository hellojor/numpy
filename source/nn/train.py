import numpy as np
import nerualnet as nl
import load_mnist as lm

dataset = lm.load_mnist()

x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']
w_list, b_list = nl.make_param([784, 100, 10])

num_epochs = 200; # 1

for epoch in range(num_epochs):
    ra = np.random.randint(60000, size=60000)
    for i in range(60):
        x_batch = x_train[ra[i*1000:(i+1)*1000], :]
        y_batch = y_train[ra[i*1000:(i+1)*1000], :]
        
        w_list, b_list = nl.update(x_batch, w_list, b_list, y_batch, lr=2.0)
    print('epoch', epoch , '.....')
    
print(y_test[0:10])

val_dict = nl.calculate(x_test, w_list, b_list)
print(val_dict['y2'][0:10].round(2))

'''
import matplotlib.pyplot as plt
plt.imshow(dataset['xtest'][8].reshape(28, 28), cmap="gray")
plt.axis("off")
plt.show()
'''