import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
from tqdm import tqdm
from librosa import display
import librosa
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Nadam, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Reshape, Lambda, BatchNormalization, Activation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix
import itertools

# #load Training & Testing Data
x_train = np.load("x_train_japan_chroma_stft.npy")
x_test = np.load("x_test_japan_chroma_stft.npy")
y_train = np.load("y_train_japan_chroma_stft.npy")
y_test = np.load("y_test_japan_chroma_stft.npy")

#converting to one hot
y_train-=1
y_test-=1
y_train = to_categorical(y_train, num_classes=420)
y_test = to_categorical(y_test, num_classes=420)
print(y_train.shape, y_test.shape)

#reshaping to 2D 
x_train=np.reshape(x_train,(x_train.shape[0], 32, 32))
x_test=np.reshape(x_test,(x_test.shape[0], 32, 32))

#reshaping to shape required by CNN
x_train=np.reshape(x_train,(x_train.shape[0], 32, 32, 1))
x_test=np.reshape(x_test,(x_test.shape[0], 32, 32, 1))
print(x_train.shape, x_test.shape)

#train_validation_split
# x_train,val_data,y_train,val_label=train_test_split(x_train,y_train,test_size=0.25,random_state=0)
# print(x_train.shape, y_train.shape, val_data.shape, val_label.shape)

optimizer = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999)

#model
model = Sequential()

#adding layers and forming the model
model.add(Conv2D(16,kernel_size=5,strides=1,padding="same",input_shape=(32,32,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(padding="same"))
model.add(Conv2D(32,kernel_size=5,strides=1,padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(padding="same"))
model.add(Conv2D(64,kernel_size=5,strides=1,padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(padding="same"))
# model.add(Conv2D(128,kernel_size=5,strides=1,padding="same"))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(padding="same"))
# model.add(Conv2D(16,kernel_size=4,strides=1,padding="same"))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(padding="same"))
model.add(Flatten())
# model.add(Dense(256,activation="relu"))
model.add(Dense(420,activation="softmax"))

model.summary()

model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])
history = model.fit(x_train,y_train,batch_size=512,epochs=90,validation_data=(x_test,y_test))

#train and test loss and scores respectively
train_loss_score=model.evaluate(x_train,y_train)
test_loss_score=model.evaluate(x_test,y_test)
print(train_loss_score)
print(test_loss_score)

# # Plot history: Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('CNN_Acc.png')


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,#这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    plt.savefig('plotplot3.jpg',dpi=350)
    plt.show()
# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val,batch_size=512)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False,target_names=labels,title='Confusion Matrix')
#=========================================================================================
#最后调用这个函数即可。 test_x是测试数据，test_y是测试标签（这里用的是One——hot向量）
#labels是一个列表，存储了你的各个类别的名字，最后会显示在横纵轴上。
#比如这里我的labels列表
labels = [*range(1,420)]

# plot_confuse(model, x_test, y_test)

predictions = model.predict_classes(x_test)
# pd.crosstab(y_test_org, predictions, rownames=['實際值'], colnames=['預測值'])