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
my_path = "/home/jerry/japanesebirdsound_plot/"

path = '/media/iml/592d21c5-889f-48cb-9a3d-eebc4000bb39/jerry/JapaneseBirdSound/'
data = pd.read_csv("JapaneseBirdSound.csv")

# y,sr = librosa.load(path, sr=None, offset=5, duration=10)

N_FFT = 1024       
HOP_SIZE = 1024       
N_MELS = 60            
WIN_SIZE = 1024      
WINDOW_TYPE = 'hann' 
FEATURE = 'mel'      
FMIN = 500

x_train = []
x_test = []

y_train = [*range(1,81)]*5
y_train = np.array(y_train)
y_test = [*range(1,361)]
y_test = np.array(y_test)

#training data
for i in tqdm(range(1,81)):
    fold_no='6'
    try:
        file=(f'0{i}-Track-0{i}.wav')
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=5, duration=3)
    
    except:
        file=(f'{i}-Track-{i}.wav')
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=5, duration=3)

    mfcc = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=20).T,axis=0)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
    
    features = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta],axis=0)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(S,ref=np.max),fmin=FMIN, fmax=sr/8)
    # # plt.colorbar(format='%+2.0f dB')
    # plt.savefig(my_path + f'{i}.png')
    
    x_train.append(features)

for i in tqdm(range(1,81)):
    fold_no='6'
    try:
        file=(f'0{i}-Track-0{i}.wav')
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=8, duration=3)
    
    except:
        file=(f'{i}-Track-{i}.wav')
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=8, duration=3)

    mfcc1 = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=20).T,axis=0)
    mfcc_delta1 = librosa.feature.delta(mfcc1)
    mfcc_delta_delta1 = librosa.feature.delta(mfcc_delta1)
    
    features1 = np.concatenate([mfcc1, mfcc_delta1, mfcc_delta_delta1],axis=0)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(S,ref=np.max),fmin=FMIN, fmax=sr/8)
    # # plt.colorbar(format='%+2.0f dB')
    # plt.savefig(my_path + f'{i}.png')
    
    x_train.append(features1)

for i in tqdm(range(1,81)):
    fold_no='6'
    try:
        file=(f'0{i}-Track-0{i}.wav')
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=11, duration=3)
    
    except:
        file=(f'{i}-Track-{i}.wav')
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=11, duration=3)

    mfcc2 = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=20).T,axis=0)
    mfcc_delta2 = librosa.feature.delta(mfcc2)
    mfcc_delta_delta2 = librosa.feature.delta(mfcc_delta2)
    
    features2 = np.concatenate([mfcc2, mfcc_delta2, mfcc_delta_delta2],axis=0)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(S,ref=np.max),fmin=FMIN, fmax=sr/8)
    # # plt.colorbar(format='%+2.0f dB')
    # plt.savefig(my_path + f'{i}.png')
    
    x_train.append(features2)

for i in tqdm(range(1,81)):
    fold_no='6'
    try:
        file=(f'0{i}-Track-0{i}.wav')
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=14, duration=3)
    
    except:
        file=(f'{i}-Track-{i}.wav')
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=14, duration=3)

    mfcc3 = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=20).T,axis=0)
    mfcc_delta3 = librosa.feature.delta(mfcc3)
    mfcc_delta_delta3 = librosa.feature.delta(mfcc_delta3)
    
    features3 = np.concatenate([mfcc3, mfcc_delta3, mfcc_delta_delta3],axis=0)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(S,ref=np.max),fmin=FMIN, fmax=sr/8)
    # # plt.colorbar(format='%+2.0f dB')
    # plt.savefig(my_path + f'{i}.png')
    
    x_train.append(features3)

for i in tqdm(range(1,81)):
    fold_no='6'
    try:
        file=(f'0{i}-Track-0{i}.wav')
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=17, duration=3)
    
    except:
        file=(f'{i}-Track-{i}.wav')
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=17, duration=3)

    mfcc4 = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=20).T,axis=0)
    mfcc_delta4 = librosa.feature.delta(mfcc4)
    mfcc_delta_delta4 = librosa.feature.delta(mfcc_delta4)
    
    features4 = np.concatenate([mfcc4, mfcc_delta4, mfcc_delta_delta4],axis=0)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(librosa.power_to_db(S,ref=np.max),fmin=FMIN, fmax=sr/8)
    # # plt.colorbar(format='%+2.0f dB')
    # plt.savefig(my_path + f'{i}.png')
    
    x_train.append(features4)

for i in tqdm(range(80,0,-1)):
    fold_no='6'
    try:
        file=(f'{i}-Track-{i}.wav')
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=23, duration=5)
    
    except:
        file=(f'0{i}-Track-0{i}.wav')
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename, sr=None, offset=23, duration=5)

    mfcc5 = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=20).T,axis=0)
    mfcc_delta5 = librosa.feature.delta(mfcc5)
    mfcc_delta_delta5 = librosa.feature.delta(mfcc_delta5)
    
    features5 = np.concatenate([mfcc5, mfcc_delta5, mfcc_delta_delta5],axis=0)

    x_test.append(features5)

x_train = np.array(x_train)
x_test = np.array(x_test)
print(x_train.shape, x_test.shape)


np.save("x_train_japan_80_divide",x_train,allow_pickle=True)
np.save("x_test_japan_80_divide",x_test,allow_pickle=True)
np.save("y_train_japan_80_divide",y_train,allow_pickle=True)

train1 = pd.DataFrame(x_train)
train2 = pd.DataFrame(x_test)
train3 = pd.DataFrame(y_train)

train1.to_csv("x_train_japan_80_divide.csv",index=False)
train2.to_csv("x_test_japan_80_divide.csv",index=False)
train3.to_csv("y_train_japan_80_divide.csv",index=False)