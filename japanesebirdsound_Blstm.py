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
x_train=np.reshape(x_train,(x_train.shape[0], 4,15))
x_test=np.reshape(x_test,(x_test.shape[0], 4,15))
input_shape = (x_train.shape[1], x_train.shape[2])
print(input_shape)

#LSTM
model = Sequential()

model.add(Bidirectional(LSTM(units=256, dropout=0.05, recurrent_dropout=0.4, return_sequences=True), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(units=256, dropout=0.05, recurrent_dropout=0.4, return_sequences=False)))
model.add(BatchNormalization())
               
# model.add(LSTM(units=512, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
# model.add(BatchNormalization())
model.add(Dense(units=420, activation="softmax"))

print("Compiling ...")
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
# SGD    : lr=0.01,  momentum=0.,                             decay=0.
opt = Adam()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 128  # num of training examples per minibatch
num_epochs = 90
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_data=(x_test,y_test)
)

train_loss, train_accuracy = model.evaluate(x_train,y_train)
test_loss, test_accuracy = model.evaluate(x_test,y_test)
print(f'train score: {train_loss}, train accuracy: {train_accuracy}')
print(f'test score: {test_loss}, test accuracy: {test_accuracy}')


# # Plot history: Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('BLSTM_Acc.png')