from tqdm import tqdm
import scipy
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
from scipy import fft, arange
import cv2
import pandas as pd
import glob
from librosa import display
import math

my_path = '/home/jerry/japanesbirdsound_hist_model/'
path = '/media/iml/jerry/JapaneseBirdSound/'
data = pd.read_csv("JapaneseBirdSound_390.csv")

for i in tqdm(range(len(data))):
    fold_no=str(data.iloc[i]["fold"])
    file=data.iloc[i]["slice_file_name"]
    label=data.iloc[i]["classID"]
    filename=path+fold_no+"/"+file
    
    y,sr = librosa.load(filename, sr=None, offset=5, duration=15)

    S = librosa.feature.mfcc(y=y,sr=sr)
    S_delta = librosa.feature.delta(S)
    S_delta_delta = librosa.feature.delta(S_delta)

    AA = np.concatenate((S, S_delta))
    AA = np.concatenate((AA, S_delta_delta))

    S = AA.astype(int)

    origin_img = np.reshape(S,(285*272))

    [m] = origin_img.shape

    h_arr = [0]*1000
    h_arr_1 = [0]*1000
    new_img = np.zeros_like(origin_img)

    #Calculate Original Histogram
    for i in range(m):
        # for j in range(n):
        h_arr[origin_img[i]]+=1

    plt.subplot(311)
    plt.bar(range(1000),h_arr)
    plt.xlabel('Gray_Scale')
    plt.ylabel('Pixel Numbers')
    plt.title('Original Histogram')

    #Transfer function
    h_arr = np.array(h_arr)*1000/(m)
    CDF = [sum(h_arr[:i+1]) for i in range(len(h_arr))]
    TF = np.float16(CDF)

    #Replace pixels with transfer function
    for i in range(m):
        # for j in range(n):
        new_img[i] = TF[origin_img[i]]

    #Calculate Equalized Histogram
    for i in range(m):
        # for j in range(n):
        h_arr_1[new_img[i]]+=1

    plt.figure(1)
    plt.subplot(312)
    plt.bar(range(1000),h_arr_1)
    plt.xlabel('Gray_Scale')
    plt.ylabel('Pixel Numbers')
    plt.title('Equalized Histogram')
    plt.subplot(313)
    plt.plot(TF)
    plt.xlabel('Gray_Scale')
    plt.ylabel('Y') 
    plt.title('Transform function')

    #Show images
    plt.savefig(my_path + f"hist_equal_{file}.png")

    print(new_img.shape)

    print(new_img.max())
    print(new_img.min())

    mfcc_list = np.reshape(new_img,())#490 450
    print(mfcc_list.shape)
