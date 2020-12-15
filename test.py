import random
from tqdm import tqdm
import time
import os
import struct
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
import numpy as np
import librosa
import librosa.display
from sklearn import manifold
import scipy, pylab
import scipy.io
from scipy.io import wavfile
from scipy.io.wavfile import write
from os.path import dirname, join as pjoin
import scipy.io as sio
import h5py
import scipy.io.wavfile as wav
from scipy import signal
import math
from scipy.fftpack import ifft2

my_path = "/home/jerry/japanesebirdsound_plot_train/"

path = '/media/iml/jerry/JapaneseBirdSound/'
data = pd.read_csv("JapaneseBirdSound.csv")

# y,sr = librosa.load(path, sr=None, offset=5, duration=10)

x_train = []
x_test = []

y_train = [*range(1,421)]*10
y_train = np.array(y_train)
y_test = [*range(1,421)]
y_test = np.array(y_test)


#training data
def generating_training_data(XX):
    for i in tqdm(range(len(data))):
        fold_no=str(data.iloc[i]["fold"])
        file=data.iloc[i]["slice_file_name"]
        label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
    
        y,sr = librosa.load(filename, sr=None, offset=XX, duration=5)

        S = np.abs(librosa.stft(y, n_fft=512, hop_length=64, window='hann'))**2
        STFT=np.mean(librosa.feature.chroma_stft(S=S, sr=sr).T,axis=1)
        STFT*=1000
        x_train.append(STFT)

#testing data
def generating_testing_data(XX):
    for i in tqdm(range(len(data))):
        fold_no=str(data.iloc[i]["fold"])
        file=data.iloc[i]["slice_file_name"]
        label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
    
        y,sr = librosa.load(filename, sr=None, offset=XX, duration=5)
    
        S = np.abs(librosa.stft(y, n_fft=512, hop_length=64, window='hann'))**2
        STFT=np.mean(librosa.feature.chroma_stft(S=S, sr=sr).T,axis=1)
        STFT*=1000
        x_test.append(STFT)


for i in range(4,14,1):
    generating_training_data(i)

generating_testing_data(18)
# generating_testing_data(21)


x_train = np.array(x_train)
x_test = np.array(x_test)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


#convert into 2D
# x_train_2d=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
# x_test_2d=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
# print(x_train_2d.shape,x_test_2d.shape)

np.save("x_train_japan_STFT",x_train,allow_pickle=True)
np.save("y_train_japan_STFT",y_train,allow_pickle=True)
np.save("x_test_japan_STFT",x_test,allow_pickle=True)
np.save("y_test_japan_STFT",y_test,allow_pickle=True)

train1 = pd.DataFrame(x_train)
train2 = pd.DataFrame(y_train)
train3 = pd.DataFrame(x_test)
train4 = pd.DataFrame(y_test)

train1.to_csv("x_train_japan_STFT.csv",index=False)
train2.to_csv("y_train_japan_STFT.csv",index=False)
train3.to_csv("x_test_japan_STFT.csv",index=False)
train4.to_csv("y_test_japan_STFT.csv",index=False)
