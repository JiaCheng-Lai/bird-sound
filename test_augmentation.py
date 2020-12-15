import os
from tqdm import tqdm
import librosa
from librosa import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as ipd
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

path = "/media/iml/jerry/JapaneseBirdSound/"
my_path = "/home/jerry/test_Augmentation/"

def Add_Noise(wav,sr,i):
    wav_n = wav + 0.009*np.random.normal(0,1,len(wav))
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(wav_n, sr=sr)
    plt.savefig(my_path + f'Noise{i}.jpg')
    librosa.output.write_wav(my_path + f'Noise{i}.wav', wav_n, sr)
    return None

def Shifting(wav,sr,i):
    wav_shift = np.roll(wav,22050)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(wav_shift, sr=sr)
    plt.savefig("Shifting.png")
    librosa.output.write_wav('Shifting.wav', wav_shift, sr)
    return None

def Stretching(wav,sr,i):
    factor = 0.4
    wav_time_stch = librosa.effects.time_stretch(wav,factor)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(wav_time_stch, sr=sr)
    plt.savefig("Stretching.png")
    librosa.output.write_wav('Stretching.wav', wav_time_stch, sr)
    return None

def Pitch_Shifting(wav,sr,i):
    wav_pitch_sf = librosa.effects.pitch_shift(wav,sr,n_steps=5)
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(wav_pitch_sf, sr=sr)
    plt.savefig(my_path + f'Pitch_+5_{i}.jpg')
    librosa.output.write_wav(my_path + f'Pitch_+5_{i}.wav', wav_pitch_sf, sr)
    return None
    
# print(wav_pitch_sf.shape)

# ipd.Audio("/media/iml/jerry/JapaneseBirdSound/1/01-Track-01.wav")

for i in tqdm(range(1,11)):
    fold_no = "1"
    try:
        file=(f'0{i}-Track-0{i}.wav')
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename)
    except:
        file=(f'{i}-Track-{i}.wav')
        # label=data.iloc[i]["classID"]
        filename=path+fold_no+"/"+file
        y,sr = librosa.load(filename)
        
    Pitch_Shifting(y,sr,i)