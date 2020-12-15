import random
from tqdm import tqdm
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import struct
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
import numpy as np
import librosa
import librosa.display
my_path = "/home/jerry/japanesebirdsound_plot/"

path = '/media/iml/jerry/JapaneseBirdSound/'
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

y_train = [*range(1,421)]*7
y_train = np.array(y_train)
y_test = [*range(1,421)]*2
y_test = np.array(y_test)

#training data
def extract_mfcc_train(offset):
    for i in tqdm(range(len(data))):
        fold_no=str(data.iloc[i]["fold"])
        file=data.iloc[i]["slice_file_name"]
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
    
        y,sr = librosa.load(filename, sr=None, offset=offset, duration=3)
        
        mfcc = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=20).T,axis=0)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
    
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta],axis=0)

        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(librosa.power_to_db(S,ref=np.max),fmin=FMIN, fmax=sr/8)
        # # plt.colorbar(format='%+2.0f dB')
        # plt.savefig(my_path + f'{i}.png')
    
        x_train.append(features)

def extract_mfcc_test(offset):
    for i in tqdm(range(len(data))):
        fold_no=str(data.iloc[i]["fold"])
        file=data.iloc[i]["slice_file_name"]
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
    
        y,sr = librosa.load(filename, sr=None, offset=offset, duration=3)

        mfcc1 = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=20).T,axis=0)
        mfcc_delta1 = librosa.feature.delta(mfcc1)
        mfcc_delta_delta1 = librosa.feature.delta(mfcc_delta1)
    
        features1 = np.concatenate([mfcc1, mfcc_delta1, mfcc_delta_delta1],axis=0)

        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(librosa.power_to_db(S,ref=np.max),fmin=FMIN, fmax=sr/8)
        # # plt.colorbar(format='%+2.0f dB')
        # plt.savefig(my_path + f'{i}.png')
    
        x_test.append(features1)

def Noise_add(offset):
    for i in tqdm(range(len(data))):
        fold_no=str(data.iloc[i]["fold"])
        file=data.iloc[i]["slice_file_name"]
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
    
        y,sr = librosa.load(filename, sr=None, offset=offset, duration=3)

        wav_n = y + 0.009*np.random.normal(0,1,len(y))
        
        mfcc = np.mean(librosa.feature.mfcc(wav_n, sr, n_mfcc=20).T,axis=0)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
    
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta],axis=0)

        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(librosa.power_to_db(S,ref=np.max),fmin=FMIN, fmax=sr/8)
        # # plt.colorbar(format='%+2.0f dB')
        # plt.savefig(my_path + f'{i}.png')
    
        x_train.append(features)

def Pitch_shift(offset):
    for i in tqdm(range(len(data))):
        fold_no=str(data.iloc[i]["fold"])
        file=data.iloc[i]["slice_file_name"]
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
    
        y,sr = librosa.load(filename, sr=None, offset=offset, duration=3)

        wav_pitch_sf = librosa.effects.pitch_shift(y,sr,n_steps=-2)
        
        mfcc = np.mean(librosa.feature.mfcc(wav_pitch_sf, sr, n_mfcc=20).T,axis=0)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
    
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta],axis=0)

        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(librosa.power_to_db(S,ref=np.max),fmin=FMIN, fmax=sr/8)
        # # plt.colorbar(format='%+2.0f dB')
        # plt.savefig(my_path + f'{i}.png')
    
        x_train.append(features)


for i in range(5,18,3):
    extract_mfcc_train(i)

Noise_add(14)
Pitch_shift(17)

extract_mfcc_test(20)
extract_mfcc_test(23)
# extract_mfcc_test(26)


x_train = np.array(x_train)
x_test = np.array(x_test)
print(x_train.shape, x_test.shape)


np.save("x_train_japan_420_divide_augmentation",x_train,allow_pickle=True)
np.save("y_train_japan_420_divide_augmentation",y_train,allow_pickle=True)
np.save("x_test_japan_420_divide_augmentation",x_test,allow_pickle=True)
np.save("y_test_japan_420_divide_augmentation",y_test,allow_pickle=True)


train1 = pd.DataFrame(x_train)
train2 = pd.DataFrame(y_train)
train3 = pd.DataFrame(x_test)
train4 = pd.DataFrame(y_test)

train1.to_csv("x_train_japan_420_divide_augmentation.csv",index=False)
train2.to_csv("y_train_japan_420_divide_augmentation.csv",index=False)
train3.to_csv("x_test_japan_420_divide_augmentaion.csv",index=False)
train4.to_csv("y_test_japan_420_divide_augmentation.csv",index=False)
