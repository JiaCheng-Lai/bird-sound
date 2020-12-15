import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense,Activation,BatchNormalization,Bidirectional,TimeDistributed,Lambda
from keras.optimizers import Nadam,Adam
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import 

# #load Training & Testing Data  
x_train = np.load("x_train_japan_420_divide.npy")  
x_test = np.load("x_test_japan_420_divide.npy")
y_train = np.load("y_train_japan_420_divide.npy")
y_test = np.load("y_test_japan_420_divide.npy")

# converting to one hot
y_train-=1
y_test-=1
y_train = to_categorical(y_train, num_classes=420)
y_test = to_categorical(y_test, num_classes=420)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#reshaping to 2D 
x_train=np.reshape(x_train,(x_train.shape[0], 12, 5))
x_test=np.reshape(x_test,(x_test.shape[0], 12, 5))

#reshaping to shape required by CNN
x_train=np.reshape(x_train,(x_train.shape[0], 12, 5, 1))
x_test=np.reshape(x_test,(x_test.shape[0], 12, 5, 1))
print(x_train.shape, x_test.shape)