import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic import logistic
from toy_shallow_nn import toy_shallow_nn
from neural_network import neural_network
from utils.datasets import load_cats

cats = os.getenv('CATS')
train_file = os.path.join(cats, 'train.h5')
test_file = os.path.join(cats, 'test.h5')
x_train, y_train, x_test, y_test, classes = load_cats(train_file, test_file)

plt.imshow(x_train[5])

# preprocess
train_flat = x_train.reshape(x_train.shape[0], -1).T
test_flat = x_test.reshape(x_test.shape[0], -1).T

train_flat = train_flat/255
test_flat = test_flat/255

# model = logistic(verbose=True)
# model = toy_shallow_nn(max_iter=5000, hidden_nodes=8, verbose=True)
model = neural_network(max_iter=3000, layers=[
                       (4, 'relu'),  (4, 'relu'), (1, 'sigmoid')], verbose=True)

model.fit(train_flat, y_train)

pred_train = model.predict(train_flat)
pred_test = model.predict(test_flat)

print("train accuracy: {} %".format(
    100 - np.mean(np.abs(pred_train - y_train)) * 100))
print("test accuracy: {} %".format(
    100 - np.mean(np.abs(pred_test - y_test)) * 100))
