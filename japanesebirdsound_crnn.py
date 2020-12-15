import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.models import Sequential
from keras.layers.recurrent import LSTM,GRU
from keras.layers import Dense,Flatten,Dropout
from keras.optimizers import Nadam,Adam,RMSprop
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Reshape, Lambda, BatchNormalization, Activation, Bidirectional

# #load Training & Testing Data 
x_train = np.load("x_train_japan_hist_equa_mfcc.npy")  
x_test = np.load("x_test_japan_hist_equa_mfcc.npy")
y_train = np.load("y_train_japan_hist_equa_mfcc.npy")
y_test = np.load("y_test_japan_hist_equa_mfcc.npy")

# converting to one hot
y_train-=1
y_test-=1
y_train = to_categorical(y_train, num_classes=420)
y_test = to_categorical(y_test, num_classes=420)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#reshaping to 2D 
x_train=np.reshape(x_train,(x_train.shape[0], 30,2))
x_test=np.reshape(x_test,(x_test.shape[0], 30,2))

#reshaping to shape required by CNN
x_train=np.reshape(x_train,(x_train.shape[0], 30,2,1))
x_test=np.reshape(x_test,(x_test.shape[0], 30,2,1))
print(x_train.shape, x_test.shape)

#CNN
model = Sequential()

#adding layers and forming the model
model.add(Conv2D(128,kernel_size=3,strides=1,padding="Same",input_shape=(30,2,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(padding="same"))
model.add(Conv2D(256,kernel_size=3,strides=1,padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(padding="same"))
model.add(Conv2D(512,kernel_size=3,strides=1,padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(padding="same"))
# model.add(Dropout(0.3))


#RNN
model.add(Reshape(target_shape=(8,960),name='reshape'))
model.add(Dense(units=512, activation="relu"))

#BLSTM
model.add(Bidirectional(LSTM(units=512, dropout=0.05, recurrent_dropout=0.4, return_sequences=False)))
model.add(BatchNormalization())

#GRU
# model.add(LSTM(units=512, dropout=0.05, recurrent_dropout=0.45, return_sequences=True))
# model.add(BatchNormalization())
# model.add(LSTM(units=512, dropout=0.05, recurrent_dropout=0.45, go_backwards=True, return_sequences=False))
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
batch_size = 512  # num of training examples per minibatch
num_epochs = 100
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
plt.savefig('CRNN_Acc.png')
